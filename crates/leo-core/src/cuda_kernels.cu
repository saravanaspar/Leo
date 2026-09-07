#include <cooperative_groups.h>


#define LEO_OUTPUTS 257
#define LEO_MAX_BLOCK_WINNERS 64
#define LEO_GLOBAL_SORT 1024
#define LEO_MAX_CONTEXT_ORDER 16
#define LEO_MAX_CONTEXT_DIM 256


struct LeoStepCounters {
    unsigned int suprathreshold;
    unsigned int block_selected;
    unsigned int global_clipped;
    unsigned int context_cells;
    unsigned int context_probes;
    unsigned int eligible_recurrent;
    unsigned int eligible_input;
    unsigned int learning_eligibility_count;
    float learning_eligibility_abs_sum;
};

struct LeoTrainingStepRecord {
    float loss;
    float neural_loss;
    float population_inhibition;
    float learning_eligibility_abs_sum;
    unsigned int predicted_index;
    unsigned int active_count;
    unsigned int suprathreshold;
    unsigned int block_selected;
    unsigned int global_clipped;
    unsigned int context_cells;
    unsigned int context_probes;
    unsigned int eligible_recurrent;
    unsigned int eligible_input;
    unsigned int learning_eligibility_count;
    unsigned int error_code;
};

__device__ __forceinline__ float leo_clamp(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ float leo_clip(float value, float limit) {
    return leo_clamp(value, -limit, limit);
}

__device__ __forceinline__ float leo_decay(float base, unsigned long long elapsed) {
    if (elapsed == 0ULL) return 1.0f;
    if (elapsed == 1ULL) return base;
    unsigned long long exponent = elapsed > 2147483647ULL ? 2147483647ULL : elapsed;
    float result = 1.0f;
    float factor = base;
    while (exponent != 0ULL) {
        if (exponent & 1ULL) result *= factor;
        exponent >>= 1ULL;
        if (exponent != 0ULL) factor *= factor;
    }
    return result;
}

__device__ __forceinline__ unsigned int leo_rotated_key(
    unsigned int neuron,
    unsigned int rotation,
    unsigned int neuron_count
) {
    return (neuron + neuron_count - rotation) % neuron_count;
}

__device__ __forceinline__ bool leo_better(
    float left_value,
    unsigned int left_neuron,
    float right_value,
    unsigned int right_neuron,
    unsigned int rotation,
    unsigned int neuron_count
) {
    if (left_value > right_value) return true;
    if (left_value < right_value) return false;
    return leo_rotated_key(left_neuron, rotation, neuron_count)
        < leo_rotated_key(right_neuron, rotation, neuron_count);
}

__device__ __forceinline__ float leo_branch_jacobian(
    unsigned char branch,
    float excitability,
    float temporal,
    float gate_raw
) {
    const float gate = leo_clamp(gate_raw, 0.0f, 1.0f);
    if (branch == 0 || branch == 1) return excitability;
    if (branch == 2) return excitability * gate;
    if (branch == 3) {
        const float derivative = (gate_raw > 0.0f && gate_raw < 1.0f) ? 1.0f : 0.0f;
        return excitability * temporal * derivative;
    }
    return 0.0f;
}

__device__ __forceinline__ float leo_surrogate(
    float margin,
    float cutoff,
    bool selected,
    float width,
    float gain
) {
    const float direct = selected && margin > 0.0f && margin < 1.0f ? 1.0f : 0.0f;
    if (width <= 0.0f || gain <= 0.0f) return direct;
    const float boundary = leo_clamp(1.0f - fabsf(margin - cutoff) / width, 0.0f, 1.0f);
    return direct + gain * boundary;
}

__device__ __forceinline__ void leo_atomic_saturating_increment(unsigned int* value) {
    unsigned int current = *value;
    while (current != 0xffffffffU) {
        const unsigned int previous = atomicCAS(value, current, current + 1U);
        if (previous == current) return;
        current = previous;
    }
}

__device__ __forceinline__ void leo_mark_changed(
    unsigned int index,
    unsigned int* marks,
    unsigned int* list,
    unsigned int* count
) {
    if (atomicCAS(&marks[index], 0U, 1U) == 0U) {
        const unsigned int position = atomicAdd(count, 1U);
        list[position] = index;
    }
}

__device__ __forceinline__ void leo_advance_recurrent_trace(
    unsigned int slot,
    unsigned long long tick,
    const LeoConfig* cfg,
    float* branch_sensitivity,
    float* membrane_sensitivity,
    float* fatigue_sensitivity,
    float* adaptation_fast_sensitivity,
    float* adaptation_medium_sensitivity,
    float* adaptation_slow_sensitivity,
    unsigned long long* last_tick
) {
    const unsigned long long previous = last_tick[slot];
    const unsigned long long elapsed = tick >= previous ? tick - previous : 0ULL;
    if (elapsed == 0ULL) return;
    branch_sensitivity[slot] *= leo_decay(cfg->branch_decay, elapsed);
    membrane_sensitivity[slot] *= leo_decay(cfg->membrane_decay, elapsed);
    fatigue_sensitivity[slot] *= leo_decay(cfg->fatigue_decay, elapsed);
    adaptation_fast_sensitivity[slot] *= leo_decay(cfg->adaptation_fast_decay, elapsed);
    adaptation_medium_sensitivity[slot] *= leo_decay(cfg->adaptation_medium_decay, elapsed);
    adaptation_slow_sensitivity[slot] *= leo_decay(cfg->adaptation_slow_decay, elapsed);
    last_tick[slot] = tick;
}

extern "C" __global__ void leo_start_tick(
    const LeoConfig* cfg,
    unsigned long long tick,
    const unsigned int* active,
    const unsigned int* active_count,
    float* activation,
    unsigned long long* touched_epoch,
    unsigned long long* learning_destination_epoch,
    LeoStepCounters* counters
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long tag = tick + 1ULL;
    if (index < cfg->neuron_count) {
        learning_destination_epoch[index] = 0ULL;
    }
    const unsigned int old_active_count = *active_count;
    if (index < old_active_count) {
        const unsigned int neuron = active[index];
        activation[neuron] = 0.0f;
        touched_epoch[neuron] = tag;
    }
    if (index == 0U) {
        counters->suprathreshold = 0U;
        counters->block_selected = 0U;
        counters->global_clipped = 0U;
        counters->context_cells = 0U;
        counters->context_probes = 0U;
        counters->eligible_recurrent = 0U;
        counters->eligible_input = 0U;
        counters->learning_eligibility_count = 0U;
        counters->learning_eligibility_abs_sum = 0.0f;
    }
}

extern "C" __global__ void leo_deliver_events(
    const LeoConfig* cfg,
    unsigned long long tick,
    const unsigned int* recurrent_target,
    const unsigned char* recurrent_branch,
    const unsigned char* recurrent_delay,
    const unsigned int* ring_count,
    const unsigned int* ring_source,
    const float* ring_activation,
    const float* ring_weight,
    float* branch_delta,
    unsigned long long* touched_epoch,
    bool learning_trace,
    float* recurrent_branch_sensitivity,
    float* recurrent_membrane_sensitivity,
    float* recurrent_fatigue_sensitivity,
    float* recurrent_adaptation_fast_sensitivity,
    float* recurrent_adaptation_medium_sensitivity,
    float* recurrent_adaptation_slow_sensitivity,
    float* recurrent_eligibility,
    unsigned long long* recurrent_last_tick,
    unsigned int* recurrent_eligible_mark,
    unsigned int* recurrent_eligible_list,
    unsigned int* recurrent_eligible_count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_delay = cfg->max_active_global * cfg->synapses_per_neuron;
    const unsigned int total = per_delay * 4U;
    if (index >= total) return;
    const unsigned int delay_index = index / per_delay;
    const unsigned int inner = index - delay_index * per_delay;
    const unsigned int record = inner / cfg->synapses_per_neuron;
    const unsigned int local_slot = inner - record * cfg->synapses_per_neuron;
    const unsigned int delay_value = delay_index == 0U ? 1U : (delay_index == 1U ? 2U : (delay_index == 2U ? 4U : 8U));
    const unsigned int bucket = (unsigned int)((tick + 9ULL - (unsigned long long)delay_value) % 9ULL);
    const unsigned int count = ring_count[bucket];
    if (record >= count) return;
    const unsigned int source_position = bucket * cfg->max_active_global + record;
    const unsigned int source = ring_source[source_position];
    const unsigned int slot = source * cfg->synapses_per_neuron + local_slot;
    if ((unsigned int)recurrent_delay[slot] != delay_value) return;

    const float presynaptic = ring_activation[source_position];
    if (learning_trace) {
        leo_advance_recurrent_trace(
            slot,
            tick,
            cfg,
            recurrent_branch_sensitivity,
            recurrent_membrane_sensitivity,
            recurrent_fatigue_sensitivity,
            recurrent_adaptation_fast_sensitivity,
            recurrent_adaptation_medium_sensitivity,
            recurrent_adaptation_slow_sensitivity,
            recurrent_last_tick
        );
        recurrent_branch_sensitivity[slot] += presynaptic;
        if (atomicCAS(&recurrent_eligible_mark[slot], 0U, 1U) == 0U) {
            const unsigned int position = atomicAdd(recurrent_eligible_count, 1U);
            recurrent_eligible_list[position] = slot;
        }
    }

    const unsigned int target = recurrent_target[slot];
    const unsigned int branch = (unsigned int)recurrent_branch[slot];
    const unsigned long long snapshot =
        ((unsigned long long)source_position * cfg->synapses_per_neuron) + local_slot;
    const float value = presynaptic * ring_weight[snapshot];
    atomicAdd(&branch_delta[(unsigned long long)target * 4ULL + branch], value);
    atomicExch(&touched_epoch[target], tick + 1ULL);
}

extern "C" __global__ void leo_inject_symbol(
    const LeoConfig* cfg,
    unsigned long long tick,
    unsigned int symbol,
    const unsigned int* input_target,
    const unsigned char* input_branch,
    const float* input_weight,
    float* branch_delta,
    unsigned long long* touched_epoch,
    bool learning_trace,
    float* input_branch_sensitivity,
    float* input_membrane_sensitivity,
    float* input_fatigue_sensitivity,
    float* input_adaptation_fast_sensitivity,
    float* input_adaptation_medium_sensitivity,
    float* input_adaptation_slow_sensitivity,
    float* input_eligibility,
    unsigned long long* input_last_tick,
    unsigned int* input_eligible_mark,
    unsigned int* input_eligible_list,
    unsigned int* input_eligible_count
) {
    const unsigned int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= cfg->input_fanout) return;
    const unsigned int slot = symbol * cfg->input_fanout + local;
    if (learning_trace) {
        leo_advance_recurrent_trace(
            slot,
            tick,
            cfg,
            input_branch_sensitivity,
            input_membrane_sensitivity,
            input_fatigue_sensitivity,
            input_adaptation_fast_sensitivity,
            input_adaptation_medium_sensitivity,
            input_adaptation_slow_sensitivity,
            input_last_tick
        );
        input_branch_sensitivity[slot] += 1.0f;
        if (atomicCAS(&input_eligible_mark[slot], 0U, 1U) == 0U) {
            const unsigned int position = atomicAdd(input_eligible_count, 1U);
            input_eligible_list[position] = slot;
        }
    }
    const unsigned int target = input_target[slot];
    const unsigned int branch = (unsigned int)input_branch[slot];
    atomicAdd(&branch_delta[(unsigned long long)target * 4ULL + branch], input_weight[slot]);
    atomicExch(&touched_epoch[target], tick + 1ULL);
}

__device__ __forceinline__ unsigned long long leo_context_key(
    const unsigned int* history,
    unsigned int history_count,
    unsigned int order
) {
    const unsigned long long FNV_PRIME = 0x100000001b3ULL;
    unsigned long long hash = 0xcbf29ce484222325ULL ^ (unsigned long long)order;
    const unsigned int start = history_count - order;
    for (unsigned int index = start; index < history_count; ++index) {
        unsigned int symbol = history[index];
        for (unsigned int byte = 0; byte < 4U; ++byte) {
            hash ^= (unsigned long long)((symbol >> (byte * 8U)) & 0xffU);
            hash *= FNV_PRIME;
        }
    }
    return hash == 0ULL ? 1ULL : hash;
}

extern "C" __global__ void leo_context_resolve(
    const LeoConfig* cfg,
    unsigned int symbol,
    bool enabled,
    bool allocate,
    unsigned int* history,
    unsigned int* history_count,
    unsigned long long* context_keys,
    float* context_embeddings,
    unsigned int* context_observations,
    unsigned int* active_context_slots,
    float* active_context_scales,
    unsigned int* active_context_count,
    unsigned int* changed_context_marks,
    unsigned int* changed_context_list,
    unsigned int* changed_context_count,
    LeoStepCounters* counters
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    unsigned int count = *history_count;
    if (count == cfg->context_max_order) {
        for (unsigned int index = 1U; index < count; ++index) history[index - 1U] = history[index];
        count -= 1U;
    }
    history[count++] = symbol;
    *history_count = count;
    *active_context_count = 0U;
    if (!enabled) return;

    const unsigned int available = count < cfg->context_max_order ? count : cfg->context_max_order;
    float scale_sum = 0.0f;
    unsigned int resolved_count = 0U;
    unsigned int probe_sum = 0U;
    const float confidence_target = (float)cfg->context_confidence_observations;

    for (unsigned int order = 1U; order <= available; ++order) {
        const unsigned long long key = leo_context_key(history, count, order);
        const unsigned int base = (order - 1U) * cfg->context_slots_per_order;
        const unsigned int start = (unsigned int)(key % (unsigned long long)cfg->context_slots_per_order);
        unsigned int empty_slot = 0xffffffffU;
        unsigned int weakest_slot = base + start;
        unsigned int weakest_observations = 0xffffffffU;
        unsigned int found_slot = 0xffffffffU;
        for (unsigned int offset = 0U; offset < cfg->context_probe_limit; ++offset) {
            const unsigned int slot = base + (start + offset) % cfg->context_slots_per_order;
            probe_sum += 1U;
            const unsigned long long existing = context_keys[slot];
            if (existing == key) {
                found_slot = slot;
                break;
            }
            if (existing == 0ULL && empty_slot == 0xffffffffU) empty_slot = slot;
            const unsigned int observations = context_observations[slot];
            if (observations < weakest_observations) {
                weakest_observations = observations;
                weakest_slot = slot;
            }
        }
        if (found_slot == 0xffffffffU) {
            if (!allocate) continue;
            found_slot = empty_slot != 0xffffffffU ? empty_slot : weakest_slot;
            if (context_keys[found_slot] != key) {
                context_keys[found_slot] = key;
                context_observations[found_slot] = 0U;
                const unsigned long long row = (unsigned long long)found_slot * cfg->context_embedding_dim;
                for (unsigned int dimension = 0U; dimension < cfg->context_embedding_dim; ++dimension) {
                    context_embeddings[row + dimension] = 0.0f;
                }
                leo_mark_changed(
                    found_slot,
                    changed_context_marks,
                    changed_context_list,
                    changed_context_count
                );
            }
        }
        const float observations = (float)context_observations[found_slot];
        const float confidence = observations / (observations + confidence_target);
        const float scale = confidence * (float)order;
        active_context_slots[resolved_count] = found_slot;
        active_context_scales[resolved_count] = scale;
        scale_sum += scale;
        resolved_count += 1U;
    }
    if (scale_sum > 0.0f) {
        for (unsigned int index = 0U; index < resolved_count; ++index) {
            active_context_scales[index] /= scale_sum;
        }
    }
    *active_context_count = resolved_count;
    counters->context_cells = resolved_count;
    counters->context_probes = probe_sum;
}

extern "C" __global__ void leo_select_blocks(
    const LeoConfig* cfg,
    unsigned long long tick,
    const float* threshold,
    const float* excitability,
    float* membrane,
    float* activation,
    float* fatigue,
    unsigned long long* refractory_until,
    float* branches,
    float* branch_delta,
    unsigned long long* branch_last_tick,
    unsigned long long* neuron_last_tick,
    float* adaptation_fast,
    float* adaptation_medium,
    float* adaptation_slow,
    const unsigned long long* touched_epoch,
    const float* population_inhibition,
    float* candidate_activation,
    unsigned int* block_winner_neuron,
    float* block_winner_value,
    float* block_cutoff,
    LeoStepCounters* counters
) {
    const unsigned int block = blockIdx.x;
    if (block >= cfg->block_count) return;
    const unsigned int lane = threadIdx.x;
    if (lane >= cfg->neurons_per_block) return;
    const unsigned int neuron = block * cfg->neurons_per_block + lane;
    const unsigned long long tag = tick + 1ULL;
    float candidate = 0.0f;
    if (neuron < cfg->neuron_count && touched_epoch[neuron] == tag) {
        const unsigned long long branch_elapsed = tick >= branch_last_tick[neuron]
            ? tick - branch_last_tick[neuron] : 0ULL;
        const unsigned long long base = (unsigned long long)neuron * 4ULL;
        if (branch_elapsed > 0ULL) {
            const float decay = leo_decay(cfg->branch_decay, branch_elapsed);
            for (unsigned int branch = 0U; branch < 4U; ++branch) branches[base + branch] *= decay;
            branch_last_tick[neuron] = tick;
        }
        for (unsigned int branch = 0U; branch < 4U; ++branch) {
            branches[base + branch] += branch_delta[base + branch];
            branch_delta[base + branch] = 0.0f;
        }

        const unsigned long long neuron_elapsed = tick >= neuron_last_tick[neuron]
            ? tick - neuron_last_tick[neuron] : 0ULL;
        if (neuron_elapsed > 0ULL) {
            membrane[neuron] *= leo_decay(cfg->membrane_decay, neuron_elapsed);
            fatigue[neuron] *= leo_decay(cfg->fatigue_decay, neuron_elapsed);
            adaptation_fast[neuron] *= leo_decay(cfg->adaptation_fast_decay, neuron_elapsed);
            adaptation_medium[neuron] *= leo_decay(cfg->adaptation_medium_decay, neuron_elapsed);
            adaptation_slow[neuron] *= leo_decay(cfg->adaptation_slow_decay, neuron_elapsed);
            neuron_last_tick[neuron] = tick;
        }

        if (refractory_until[neuron] <= tick) {
            const float gate = leo_clamp(branches[base + 3ULL], 0.0f, 1.0f);
            const float evidence = branches[base] + branches[base + 2ULL] * gate + branches[base + 1ULL];
            const float adaptation = adaptation_fast[neuron] + adaptation_medium[neuron] + adaptation_slow[neuron];
            const float drive = excitability[neuron] * evidence - fatigue[neuron] - adaptation - *population_inhibition;
            membrane[neuron] += drive;
            candidate = leo_clamp(membrane[neuron] - threshold[neuron], 0.0f, 1.0f);
            if (candidate > 0.0f) atomicAdd(&counters->suprathreshold, 1U);
        }
    }
    candidate_activation[neuron] = candidate;
    __syncthreads();

    if (lane == 0U) {
        const unsigned int rotation = (unsigned int)(tick % (unsigned long long)cfg->neuron_count);
        const unsigned int keep = cfg->max_active_per_block;
        float top_value[LEO_MAX_BLOCK_WINNERS + 1];
        unsigned int top_neuron[LEO_MAX_BLOCK_WINNERS + 1];
        for (unsigned int i = 0U; i <= keep; ++i) {
            top_value[i] = -1.0f;
            top_neuron[i] = 0U;
        }
        unsigned int positive = 0U;
        const unsigned int start = block * cfg->neurons_per_block;
        for (unsigned int local = 0U; local < cfg->neurons_per_block; ++local) {
            const unsigned int candidate_neuron = start + local;
            const float value = candidate_activation[candidate_neuron];
            if (value <= 0.0f) continue;
            positive += 1U;
            unsigned int position = keep;
            for (unsigned int scan = 0U; scan <= keep; ++scan) {
                if (top_value[scan] < 0.0f || leo_better(
                    value,
                    candidate_neuron,
                    top_value[scan],
                    top_neuron[scan],
                    rotation,
                    cfg->neuron_count
                )) {
                    position = scan;
                    break;
                }
            }
            if (position <= keep) {
                for (unsigned int move = keep; move > position; --move) {
                    top_value[move] = top_value[move - 1U];
                    top_neuron[move] = top_neuron[move - 1U];
                }
                top_value[position] = value;
                top_neuron[position] = candidate_neuron;
            }
        }
        const unsigned int winner_count = positive < keep ? positive : keep;
        const unsigned int winner_base = block * keep;
        for (unsigned int i = 0U; i < keep; ++i) {
            if (i < winner_count) {
                block_winner_neuron[winner_base + i] = top_neuron[i];
                block_winner_value[winner_base + i] = top_value[i];
            } else {
                block_winner_neuron[winner_base + i] = 0U;
                block_winner_value[winner_base + i] = -1.0f;
            }
        }
        block_cutoff[block] = positive > keep ? top_value[keep] : 0.0f;
        atomicAdd(&counters->block_selected, winner_count);
    }
}

extern "C" __global__ void leo_select_global(
    const LeoConfig* cfg,
    unsigned long long tick,
    const unsigned int* block_winner_neuron,
    const float* block_winner_value,
    unsigned int* active,
    float* active_value,
    unsigned int* active_count,
    float* activation,
    unsigned long long* selected_epoch,
    unsigned long long* learning_destination_epoch,
    float* population_cutoff,
    float* population_inhibition,
    LeoStepCounters* counters
) {
    __shared__ float values[LEO_GLOBAL_SORT];
    __shared__ unsigned int neurons[LEO_GLOBAL_SORT];
    const unsigned int lane = threadIdx.x;
    const unsigned int available = cfg->block_count * cfg->max_active_per_block;
    if (lane < available) {
        values[lane] = block_winner_value[lane];
        neurons[lane] = block_winner_neuron[lane];
    } else {
        values[lane] = -1.0f;
        neurons[lane] = 0U;
    }
    __syncthreads();

    const unsigned int rotation = (unsigned int)(tick % (unsigned long long)cfg->neuron_count);
    for (unsigned int width = 2U; width <= LEO_GLOBAL_SORT; width <<= 1U) {
        for (unsigned int stride = width >> 1U; stride > 0U; stride >>= 1U) {
            const unsigned int other = lane ^ stride;
            if (other > lane) {
                const bool better_first = (lane & width) == 0U;
                const bool left_better = leo_better(
                    values[lane], neurons[lane], values[other], neurons[other], rotation, cfg->neuron_count
                );
                const bool should_swap = better_first ? !left_better : left_better;
                if (should_swap) {
                    const float value = values[lane];
                    values[lane] = values[other];
                    values[other] = value;
                    const unsigned int neuron = neurons[lane];
                    neurons[lane] = neurons[other];
                    neurons[other] = neuron;
                }
            }
            __syncthreads();
        }
    }

    if (lane == 0U) {
        unsigned int selected = 0U;
        while (selected < available && selected < LEO_GLOBAL_SORT && values[selected] > 0.0f) selected += 1U;
        const unsigned int total = selected;
        const unsigned int cap = cfg->max_active_global;
        const unsigned int kept = total < cap ? total : cap;
        *active_count = kept;
        *population_cutoff = total > cap ? leo_clamp(values[cap], 0.0f, cfg->population_inhibition_max) : 0.0f;
        counters->global_clipped = total > cap ? total - cap : 0U;
        const float activity = (float)kept / (float)(cfg->neuron_count == 0U ? 1U : cfg->neuron_count);
        const float error = activity - cfg->target_activity;
        *population_inhibition = leo_clamp(
            *population_inhibition + cfg->population_inhibition_rate * error,
            0.0f,
            cfg->population_inhibition_max
        );
        for (unsigned int index = 0U; index < kept; ++index) {
            const unsigned int neuron = neurons[index];
            active[index] = neuron;
            active_value[index] = values[index];
            activation[neuron] = values[index];
            selected_epoch[neuron] = tick + 1ULL;
            learning_destination_epoch[neuron] = tick + 1ULL;
        }
    }
}

extern "C" __global__ void leo_cache_surrogate(
    const LeoConfig* cfg,
    unsigned long long tick,
    const float* threshold,
    const float* membrane,
    const unsigned long long* refractory_until,
    const unsigned long long* touched_epoch,
    const unsigned long long* selected_epoch,
    const float* block_cutoff,
    const float* population_cutoff,
    float* surrogate
) {
    const unsigned int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= cfg->neuron_count) return;
    const unsigned long long tag = tick + 1ULL;
    if (touched_epoch[neuron] != tag || refractory_until[neuron] > tick) {
        surrogate[neuron] = 0.0f;
        return;
    }
    const float margin = membrane[neuron] - threshold[neuron];
    const unsigned int block = neuron / cfg->neurons_per_block;
    const float cutoff = block_cutoff[block] > *population_cutoff ? block_cutoff[block] : *population_cutoff;
    surrogate[neuron] = leo_surrogate(
        margin,
        cutoff,
        selected_epoch[neuron] == tag,
        cfg->surrogate_width,
        cfg->surrogate_gain
    );
}

extern "C" __global__ void leo_update_recurrent_eligibility(
    const LeoConfig* cfg,
    unsigned long long tick,
    const unsigned int* recurrent_target,
    const unsigned char* recurrent_branch,
    const float* excitability,
    const float* branches,
    const unsigned long long* refractory_until,
    const unsigned long long* touched_epoch,
    const unsigned long long* selected_epoch,
    const float* surrogate,
    float* branch_sensitivity,
    float* membrane_sensitivity,
    float* fatigue_sensitivity,
    float* adaptation_fast_sensitivity,
    float* adaptation_medium_sensitivity,
    float* adaptation_slow_sensitivity,
    float* eligibility,
    unsigned long long* last_tick,
    unsigned int* eligible_mark,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    unsigned int* next_eligible_list,
    unsigned int* next_eligible_count,
    unsigned long long* learning_destination_epoch,
    LeoStepCounters* counters
) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        if (eligible_mark[slot] == 0U) continue;
        leo_advance_recurrent_trace(
            slot, tick, cfg,
            branch_sensitivity, membrane_sensitivity, fatigue_sensitivity,
            adaptation_fast_sensitivity, adaptation_medium_sensitivity, adaptation_slow_sensitivity,
            last_tick
        );
        const unsigned int destination = recurrent_target[slot];
        const unsigned long long tag = tick + 1ULL;
        float e;
        if (touched_epoch[destination] != tag || refractory_until[destination] > tick) {
            e = eligibility[slot] * cfg->eligibility_decay;
        } else {
            const unsigned long long base = (unsigned long long)destination * 4ULL;
            const float jacobian = leo_branch_jacobian(
                recurrent_branch[slot],
                excitability[destination],
                branches[base + 2ULL],
                branches[base + 3ULL]
            );
            const float adaptation_sum = adaptation_fast_sensitivity[slot]
                + adaptation_medium_sensitivity[slot] + adaptation_slow_sensitivity[slot];
            const float membrane_value = membrane_sensitivity[slot]
                + jacobian * branch_sensitivity[slot]
                - fatigue_sensitivity[slot] - adaptation_sum;
            membrane_sensitivity[slot] = membrane_value;
            e = surrogate[destination] * membrane_value;
        }
        eligibility[slot] = e;
        if (touched_epoch[destination] == tag && selected_epoch[destination] == tag) {
            fatigue_sensitivity[slot] += cfg->fatigue_gain * e;
            adaptation_fast_sensitivity[slot] += cfg->adaptation_fast_gain * e;
            adaptation_medium_sensitivity[slot] += cfg->adaptation_medium_gain * e;
            adaptation_slow_sensitivity[slot] += cfg->adaptation_slow_gain * e;
        }
        float magnitude = fabsf(branch_sensitivity[slot]);
        magnitude = fmaxf(magnitude, fabsf(membrane_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(fatigue_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(adaptation_fast_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(adaptation_medium_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(adaptation_slow_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(e));
        if (magnitude > cfg->eligibility_epsilon) {
            const unsigned int next = atomicAdd(next_eligible_count, 1U);
            next_eligible_list[next] = slot;
            atomicAdd(&counters->eligible_recurrent, 1U);
            atomicExch(&learning_destination_epoch[destination], tag);
        } else {
            eligible_mark[slot] = 0U;
        }
    }
}

extern "C" __global__ void leo_update_input_eligibility(
    const LeoConfig* cfg,
    unsigned long long tick,
    const unsigned int* input_target,
    const unsigned char* input_branch,
    const float* excitability,
    const float* branches,
    const unsigned long long* refractory_until,
    const unsigned long long* touched_epoch,
    const unsigned long long* selected_epoch,
    const float* surrogate,
    float* branch_sensitivity,
    float* membrane_sensitivity,
    float* fatigue_sensitivity,
    float* adaptation_fast_sensitivity,
    float* adaptation_medium_sensitivity,
    float* adaptation_slow_sensitivity,
    float* eligibility,
    unsigned long long* last_tick,
    unsigned int* eligible_mark,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    unsigned int* next_eligible_list,
    unsigned int* next_eligible_count,
    unsigned long long* learning_destination_epoch,
    LeoStepCounters* counters
) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        if (eligible_mark[slot] == 0U) continue;
        leo_advance_recurrent_trace(
            slot, tick, cfg,
            branch_sensitivity, membrane_sensitivity, fatigue_sensitivity,
            adaptation_fast_sensitivity, adaptation_medium_sensitivity, adaptation_slow_sensitivity,
            last_tick
        );
        const unsigned int destination = input_target[slot];
        const unsigned long long tag = tick + 1ULL;
        float e;
        if (touched_epoch[destination] != tag || refractory_until[destination] > tick) {
            e = eligibility[slot] * cfg->eligibility_decay;
        } else {
            const unsigned long long base = (unsigned long long)destination * 4ULL;
            const float jacobian = leo_branch_jacobian(
                input_branch[slot],
                excitability[destination],
                branches[base + 2ULL],
                branches[base + 3ULL]
            );
            const float adaptation_sum = adaptation_fast_sensitivity[slot]
                + adaptation_medium_sensitivity[slot] + adaptation_slow_sensitivity[slot];
            const float membrane_value = membrane_sensitivity[slot]
                + jacobian * branch_sensitivity[slot]
                - fatigue_sensitivity[slot] - adaptation_sum;
            membrane_sensitivity[slot] = membrane_value;
            e = surrogate[destination] * membrane_value;
        }
        eligibility[slot] = e;
        if (touched_epoch[destination] == tag && selected_epoch[destination] == tag) {
            fatigue_sensitivity[slot] += cfg->fatigue_gain * e;
            adaptation_fast_sensitivity[slot] += cfg->adaptation_fast_gain * e;
            adaptation_medium_sensitivity[slot] += cfg->adaptation_medium_gain * e;
            adaptation_slow_sensitivity[slot] += cfg->adaptation_slow_gain * e;
        }
        float magnitude = fabsf(branch_sensitivity[slot]);
        magnitude = fmaxf(magnitude, fabsf(membrane_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(fatigue_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(adaptation_fast_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(adaptation_medium_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(adaptation_slow_sensitivity[slot]));
        magnitude = fmaxf(magnitude, fabsf(e));
        if (magnitude > cfg->eligibility_epsilon) {
            const unsigned int next = atomicAdd(next_eligible_count, 1U);
            next_eligible_list[next] = slot;
            atomicAdd(&counters->eligible_input, 1U);
            atomicExch(&learning_destination_epoch[destination], tag);
        } else {
            eligible_mark[slot] = 0U;
        }
    }
}

extern "C" __global__ void leo_post_and_emit(
    const LeoConfig* cfg,
    unsigned long long tick,
    const float* threshold,
    const float* recurrent_weight,
    const unsigned int* active,
    const float* active_value,
    const unsigned int* active_count,
    float* membrane,
    float* fatigue,
    unsigned long long* refractory_until,
    float* adaptation_fast,
    float* adaptation_medium,
    float* adaptation_slow,
    unsigned int* ring_count,
    unsigned int* ring_source,
    float* ring_activation,
    float* ring_weight
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = *active_count;
    const unsigned int bucket = (unsigned int)(tick % 9ULL);
    if (index == 0U) ring_count[bucket] = count;
    if (index >= count) return;
    const unsigned int neuron = active[index];
    const float value = active_value[index];
    fatigue[neuron] += cfg->fatigue_gain * value;
    adaptation_fast[neuron] += cfg->adaptation_fast_gain * value;
    adaptation_medium[neuron] += cfg->adaptation_medium_gain * value;
    adaptation_slow[neuron] += cfg->adaptation_slow_gain * value;
    membrane[neuron] -= threshold[neuron] * cfg->membrane_reset_fraction;
    refractory_until[neuron] = tick + (unsigned long long)cfg->refractory_ticks + 1ULL;

    const unsigned int ring_position = bucket * cfg->max_active_global + index;
    ring_source[ring_position] = neuron;
    ring_activation[ring_position] = value;
    const unsigned long long ring_base = (unsigned long long)ring_position * cfg->synapses_per_neuron;
    const unsigned long long source_base = (unsigned long long)neuron * cfg->synapses_per_neuron;
    for (unsigned int local = 0U; local < cfg->synapses_per_neuron; ++local) {
        ring_weight[ring_base + local] = recurrent_weight[source_base + local];
    }
}

extern "C" __global__ void leo_forward(
    const LeoConfig* cfg,
    const float* output_weights,
    const float* output_bias,
    const float* context_embeddings,
    const float* context_output_weights,
    const unsigned int* active,
    const float* active_value,
    const unsigned int* active_count,
    const unsigned int* context_slots,
    const float* context_scales,
    const unsigned int* context_count,
    int target_index,
    float* neural_logits,
    float* logits,
    float* probabilities,
    float* errors,
    float* context_latent,
    float* context_gradient
) {
    __shared__ float latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float reduction[512];
    const unsigned int lane = threadIdx.x;
    const unsigned int embedding_dim = cfg->context_embedding_dim;
    const unsigned int contexts = *context_count;
    if (lane < embedding_dim) {
        float value = 0.0f;
        for (unsigned int index = 0U; index < contexts; ++index) {
            const unsigned int slot = context_slots[index];
            value += context_embeddings[(unsigned long long)slot * embedding_dim + lane] * context_scales[index];
        }
        latent[lane] = value;
        context_latent[lane] = value;
    }
    __syncthreads();

    float combined = -3.402823466e+38F;
    if (lane < LEO_OUTPUTS) {
        float neural = output_bias[lane];
        const unsigned int count = *active_count;
        for (unsigned int index = 0U; index < count; ++index) {
            const unsigned int neuron = active[index];
            neural += output_weights[(unsigned long long)neuron * LEO_OUTPUTS + lane] * active_value[index];
        }
        combined = neural;
        const unsigned long long context_row = (unsigned long long)lane * embedding_dim;
        for (unsigned int dimension = 0U; dimension < embedding_dim; ++dimension) {
            combined += context_output_weights[context_row + dimension] * latent[dimension];
        }
        neural_logits[lane] = neural;
        logits[lane] = combined;
    }
    reduction[lane] = combined;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];
    const float exponential = lane < LEO_OUTPUTS ? expf(combined - maximum) : 0.0f;
    reduction[lane] = exponential;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    const float sum = reduction[0];
    if (lane < LEO_OUTPUTS) {
        const bool valid = sum == sum && sum > 0.0f && sum < 3.402823466e+38F;
        const float probability = valid ? exponential / sum : 1.0f / (float)LEO_OUTPUTS;
        probabilities[lane] = probability;
        errors[lane] = target_index >= 0 ? probability - (lane == (unsigned int)target_index ? 1.0f : 0.0f) : 0.0f;
    }
    __syncthreads();
    if (lane < embedding_dim) {
        float gradient = 0.0f;
        if (target_index >= 0) {
            for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) {
                gradient += context_output_weights[(unsigned long long)output * embedding_dim + lane] * errors[output];
            }
        }
        context_gradient[lane] = gradient;
    }
}

extern "C" __global__ void leo_capture_training_step(
    const float* probabilities,
    const float* neural_logits,
    const unsigned int* active_count,
    const float* population_inhibition,
    const LeoStepCounters* counters,
    unsigned int* error_flag,
    int target_index,
    LeoTrainingStepRecord* records,
    unsigned int record_index
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    LeoTrainingStepRecord record;
    unsigned int predicted = 0U;
    float best = probabilities[0];
    for (unsigned int output = 1U; output < LEO_OUTPUTS; ++output) {
        const float candidate = probabilities[output];
        if (candidate > best) {
            best = candidate;
            predicted = output;
        }
    }

    float loss = 0.0f;
    float neural_loss = 0.0f;
    if (target_index >= 0) {
        const float probability = fmaxf(probabilities[(unsigned int)target_index], 1.0e-12f);
        loss = -logf(probability);
        float maximum = neural_logits[0];
        for (unsigned int output = 1U; output < LEO_OUTPUTS; ++output) {
            maximum = fmaxf(maximum, neural_logits[output]);
        }
        float sum = 0.0f;
        for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) {
            sum += expf(neural_logits[output] - maximum);
        }
        neural_loss = (sum == sum && sum > 0.0f && sum < 3.402823466e+38F)
            ? maximum + logf(sum) - neural_logits[(unsigned int)target_index]
            : logf((float)LEO_OUTPUTS);
    }

    record.loss = loss;
    record.neural_loss = neural_loss;
    record.population_inhibition = *population_inhibition;
    record.learning_eligibility_abs_sum = counters->learning_eligibility_abs_sum;
    record.predicted_index = predicted;
    record.active_count = *active_count;
    record.suprathreshold = counters->suprathreshold;
    record.block_selected = counters->block_selected;
    record.global_clipped = counters->global_clipped;
    record.context_cells = counters->context_cells;
    record.context_probes = counters->context_probes;
    record.eligible_recurrent = counters->eligible_recurrent;
    record.eligible_input = counters->eligible_input;
    record.learning_eligibility_count = counters->learning_eligibility_count;
    record.error_code = *error_flag;
    *error_flag = 0U;
    records[record_index] = record;
}

extern "C" __global__ void leo_learning_signals(
    const LeoConfig* cfg,
    unsigned long long tick,
    const float* output_weights,
    const float* errors,
    const unsigned long long* learning_destination_epoch,
    float* learning_signal
) {
    const unsigned int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= cfg->neuron_count) return;
    if (learning_destination_epoch[neuron] != tick + 1ULL) {
        learning_signal[neuron] = 0.0f;
        return;
    }
    const unsigned long long row = (unsigned long long)neuron * LEO_OUTPUTS;
    float signal = 0.0f;
    for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) signal += output_weights[row + output] * errors[output];
    learning_signal[neuron] = signal;
}

extern "C" __global__ void leo_update_output(
    const LeoConfig* cfg,
    float* output_weights,
    float* output_bias,
    const unsigned int* active,
    const float* active_value,
    const unsigned int* active_count,
    const float* errors,
    float supervised_strength,
    unsigned int* changed_output_marks,
    unsigned int* changed_output_list,
    unsigned int* changed_output_count,
    unsigned int* error_flag
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = *active_count;
    const unsigned int weight_count = count * LEO_OUTPUTS;
    if (index < weight_count) {
        const unsigned int active_index = index / LEO_OUTPUTS;
        const unsigned int output = index - active_index * LEO_OUTPUTS;
        const unsigned int neuron = active[active_index];
        const unsigned long long slot = (unsigned long long)neuron * LEO_OUTPUTS + output;
        const float delta = leo_clip(
            -cfg->output_learning_rate * supervised_strength * errors[output] * active_value[active_index],
            cfg->max_update
        );
        const float updated = leo_clamp(output_weights[slot] + delta, cfg->weight_min, cfg->weight_max);
        if (!isfinite(updated)) atomicCAS(error_flag, 0U, 1U);
        else output_weights[slot] = updated;
        if (output == 0U) leo_mark_changed(neuron, changed_output_marks, changed_output_list, changed_output_count);
        return;
    }
    const unsigned int bias_index = index - weight_count;
    if (bias_index < LEO_OUTPUTS) {
        const float delta = leo_clip(
            -cfg->output_learning_rate * supervised_strength * errors[bias_index], cfg->max_update
        );
        const float updated = output_bias[bias_index] + delta;
        if (!isfinite(updated)) atomicCAS(error_flag, 0U, 2U);
        else output_bias[bias_index] = updated;
    }
}

extern "C" __global__ void leo_update_context(
    const LeoConfig* cfg,
    float* context_output_weights,
    float* context_embeddings,
    unsigned int* context_observations,
    const unsigned int* context_slots,
    const float* context_scales,
    const unsigned int* context_count,
    const float* context_latent,
    const float* context_gradient,
    const float* errors,
    float supervised_strength,
    unsigned int* changed_context_marks,
    unsigned int* changed_context_list,
    unsigned int* changed_context_count,
    unsigned int* error_flag
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int embedding_dim = cfg->context_embedding_dim;
    const unsigned int projection_count = LEO_OUTPUTS * embedding_dim;
    if (index < projection_count) {
        const unsigned int output = index / embedding_dim;
        const unsigned int dimension = index - output * embedding_dim;
        const float gradient = errors[output] * context_latent[dimension];
        const float delta = leo_clip(-cfg->context_learning_rate * supervised_strength * gradient, cfg->max_update);
        const float updated = leo_clamp(context_output_weights[index] + delta, cfg->weight_min, cfg->weight_max);
        if (!isfinite(updated)) atomicCAS(error_flag, 0U, 3U);
        else context_output_weights[index] = updated;
        return;
    }
    const unsigned int embedding_index = index - projection_count;
    const unsigned int contexts = *context_count;
    const unsigned int embedding_count = contexts * embedding_dim;
    if (embedding_index < embedding_count) {
        const unsigned int context_index = embedding_index / embedding_dim;
        const unsigned int dimension = embedding_index - context_index * embedding_dim;
        const unsigned int slot = context_slots[context_index];
        const unsigned long long target = (unsigned long long)slot * embedding_dim + dimension;
        const float gradient = context_scales[context_index] * context_gradient[dimension];
        const float delta = leo_clip(-cfg->context_learning_rate * supervised_strength * gradient, cfg->max_update);
        const float updated = leo_clamp(context_embeddings[target] + delta, cfg->weight_min, cfg->weight_max);
        if (!isfinite(updated)) atomicCAS(error_flag, 0U, 4U);
        else context_embeddings[target] = updated;
        if (dimension == 0U) {
            leo_mark_changed(slot, changed_context_marks, changed_context_list, changed_context_count);
            leo_atomic_saturating_increment(&context_observations[slot]);
        }
    }
}

extern "C" __global__ void leo_update_recurrent_weights(
    const LeoConfig* cfg,
    const unsigned int* recurrent_target,
    const unsigned char* neuron_type,
    float* recurrent_weight,
    const float* eligibility,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    const float* learning_signal,
    float supervised_strength,
    unsigned int* changed_marks,
    unsigned int* changed_list,
    unsigned int* changed_count,
    LeoStepCounters* counters,
    unsigned int* error_flag
) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        const unsigned int destination = recurrent_target[slot];
        const float e = eligibility[slot];
        const float gradient = learning_signal[destination] * e;
        const float delta = leo_clip(-cfg->recurrent_learning_rate * supervised_strength * gradient, cfg->max_update);
        const float old_weight = recurrent_weight[slot];
        float updated = leo_clamp(old_weight + delta, cfg->weight_min, cfg->weight_max);
        const unsigned int source = slot / cfg->synapses_per_neuron;
        updated = neuron_type[source] == 0U ? fmaxf(updated, 0.0f) : fminf(updated, 0.0f);
        if (!isfinite(updated)) {
            atomicCAS(error_flag, 0U, 5U);
            continue;
        }
        if (updated != old_weight) {
            recurrent_weight[slot] = updated;
            leo_mark_changed(slot, changed_marks, changed_list, changed_count);
        }
        atomicAdd(&counters->learning_eligibility_abs_sum, fabsf(e));
        atomicAdd(&counters->learning_eligibility_count, 1U);
    }
}

extern "C" __global__ void leo_update_input_weights(
    const LeoConfig* cfg,
    const unsigned int* input_target,
    float* input_weight,
    const float* eligibility,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    const float* learning_signal,
    float supervised_strength,
    unsigned int* changed_marks,
    unsigned int* changed_list,
    unsigned int* changed_count,
    LeoStepCounters* counters,
    unsigned int* error_flag
) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        const unsigned int destination = input_target[slot];
        const float e = eligibility[slot];
        const float gradient = learning_signal[destination] * e;
        const float delta = leo_clip(-cfg->recurrent_learning_rate * supervised_strength * gradient, cfg->max_update);
        const float old_weight = input_weight[slot];
        const float updated = leo_clamp(old_weight + delta, 0.0f, cfg->weight_max);
        if (!isfinite(updated)) {
            atomicCAS(error_flag, 0U, 6U);
            continue;
        }
        if (updated != old_weight) {
            input_weight[slot] = updated;
            leo_mark_changed(slot, changed_marks, changed_list, changed_count);
        }
        atomicAdd(&counters->learning_eligibility_abs_sum, fabsf(e));
        atomicAdd(&counters->learning_eligibility_count, 1U);
    }
}

extern "C" __global__ void leo_inhibitory_homeostasis(
    const LeoConfig* cfg,
    const unsigned int* active,
    const float* active_value,
    const unsigned int* active_count,
    const float* activation,
    const unsigned char* neuron_type,
    const unsigned int* recurrent_target,
    float* recurrent_weight,
    float strength,
    unsigned int* changed_marks,
    unsigned int* changed_list,
    unsigned int* changed_count,
    unsigned int* error_flag
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = *active_count;
    const unsigned int work = count * cfg->synapses_per_neuron;
    if (index >= work) return;
    const unsigned int active_index = index / cfg->synapses_per_neuron;
    const unsigned int local = index - active_index * cfg->synapses_per_neuron;
    const unsigned int source = active[active_index];
    if (neuron_type[source] == 0U) return;
    const unsigned int slot = source * cfg->synapses_per_neuron + local;
    const unsigned int target = recurrent_target[slot];
    const float target_activity = activation[target] > 0.0f ? 1.0f : 0.0f;
    const float activity_error = target_activity - cfg->target_activity;
    const float delta = leo_clip(
        -cfg->inhibitory_learning_rate * strength * active_value[active_index] * activity_error,
        cfg->max_update
    );
    const float old_weight = recurrent_weight[slot];
    const float updated = leo_clamp(old_weight + delta, cfg->weight_min, 0.0f);
    if (!isfinite(updated)) {
        atomicCAS(error_flag, 0U, 7U);
        return;
    }
    if (updated != old_weight) {
        recurrent_weight[slot] = updated;
        leo_mark_changed(slot, changed_marks, changed_list, changed_count);
    }
}

extern "C" __global__ void leo_homeostasis(
    const LeoConfig* cfg,
    unsigned long long tick,
    const unsigned int* active_count,
    const float* activation,
    const unsigned long long* touched_epoch,
    float* threshold,
    unsigned int* changed_marks,
    unsigned int* changed_list,
    unsigned int* changed_count
) {
    const unsigned int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= cfg->neuron_count || touched_epoch[neuron] != tick + 1ULL) return;
    const float population_activity = (float)(*active_count) / (float)(cfg->neuron_count == 0U ? 1U : cfg->neuron_count);
    const float population_error = population_activity - cfg->target_activity;
    const float local_activity = activation[neuron] > 0.0f ? 1.0f : 0.0f;
    const float local_error = local_activity - cfg->target_activity;
    const float old_threshold = threshold[neuron];
    const float updated = leo_clamp(
        old_threshold + cfg->threshold_homeostasis_rate * (local_error + population_error),
        0.05f,
        2.0f
    );
    if (updated != old_threshold) {
        threshold[neuron] = updated;
        leo_mark_changed(neuron, changed_marks, changed_list, changed_count);
    }
}

// Persistent document training path. The non-cooperative per-step kernels above remain
// the correctness reference; this path executes the same phases inside one
// cooperative launch and synchronizes the whole grid between phases.

struct LeoPersistentStep {
    unsigned int symbol;
    int target_index;
    unsigned int context_enabled;
    float supervised_strength;
};


template <typename T>
__device__ __forceinline__ T* leo_p_ptr(const unsigned long long* pointers, unsigned int index) {
    return reinterpret_cast<T*>(pointers[index]);
}

template <typename T>
__device__ __forceinline__ const T* leo_p_cptr(const unsigned long long* pointers, unsigned int index) {
    return reinterpret_cast<const T*>(pointers[index]);
}

// Persistent story execution is block-local: one CUDA block owns one story.
// This deliberately trades intra-story grid parallelism for independent story
// parallelism, eliminating grid-wide barriers and host launches per byte.
__device__ __forceinline__ unsigned int leo_p_global_thread() {
    return threadIdx.x;
}

__device__ __forceinline__ unsigned int leo_p_global_stride() {
    return blockDim.x;
}


// Exact per-tick worklists. Epoch arrays remain the source of truth for
// membership; the lists only materialize the unique touched/destination set so
// later phases do not scan every neuron.
__device__ __forceinline__ void leo_p_mark_touched(
    const unsigned long long* p,
    unsigned int neuron,
    unsigned long long tag
) {
    unsigned long long* epoch = leo_p_ptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    if (atomicExch(&epoch[neuron], tag) != tag) {
        unsigned int* count = leo_p_ptr<unsigned int>(p, LEO_P_TOUCHED_COUNT);
        unsigned int* list = leo_p_ptr<unsigned int>(p, LEO_P_TOUCHED_LIST);
        const unsigned int position = atomicAdd(count, 1U);
        list[position] = neuron;
    }
}

__device__ __forceinline__ void leo_p_mark_learning_destination(
    const unsigned long long* p,
    unsigned int neuron,
    unsigned long long tag
) {
    unsigned long long* epoch = leo_p_ptr<unsigned long long>(p, LEO_P_LEARNING_DESTINATION_EPOCH);
    if (atomicExch(&epoch[neuron], tag) != tag) {
        unsigned int* count = leo_p_ptr<unsigned int>(p, LEO_P_LEARNING_DESTINATION_COUNT);
        unsigned int* list = leo_p_ptr<unsigned int>(p, LEO_P_LEARNING_DESTINATION_LIST);
        const unsigned int position = atomicAdd(count, 1U);
        list[position] = neuron;
    }
}

__device__ void leo_p_start_tick(const unsigned long long* p, unsigned long long tick) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    float* activation = leo_p_ptr<float>(p, LEO_P_ACTIVATION);
    unsigned int* touched_count = leo_p_ptr<unsigned int>(p, LEO_P_TOUCHED_COUNT);
    unsigned int* destination_count = leo_p_ptr<unsigned int>(p, LEO_P_LEARNING_DESTINATION_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();
    const unsigned long long tag = tick + 1ULL;
    const unsigned int old_active_count = *active_count;
    if (thread == 0U) {
        *touched_count = 0U;
        *destination_count = 0U;
    }
    __syncthreads();
    for (unsigned int index = thread; index < old_active_count; index += stride) {
        const unsigned int neuron = active[index];
        activation[neuron] = 0.0f;
        leo_p_mark_touched(p, neuron, tag);
    }
    if (thread == 0U) {
        counters->suprathreshold = 0U;
        counters->block_selected = 0U;
        counters->global_clipped = 0U;
        counters->context_cells = 0U;
        counters->context_probes = 0U;
        counters->eligible_recurrent = 0U;
        counters->eligible_input = 0U;
        counters->learning_eligibility_count = 0U;
        counters->learning_eligibility_abs_sum = 0.0f;
    }
}

__device__ void leo_p_deliver_events(
    const unsigned long long* p,
    unsigned long long tick,
    bool learning_trace,
    unsigned int* rec_list,
    unsigned int* rec_count
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* recurrent_target = leo_p_cptr<unsigned int>(p, LEO_P_RECURRENT_TARGET);
    const unsigned char* recurrent_branch = leo_p_cptr<unsigned char>(p, LEO_P_RECURRENT_BRANCH);
    const unsigned char* recurrent_delay = leo_p_cptr<unsigned char>(p, LEO_P_RECURRENT_DELAY);
    const unsigned int* ring_count = leo_p_cptr<unsigned int>(p, LEO_P_RING_COUNT);
    const unsigned int* ring_source = leo_p_cptr<unsigned int>(p, LEO_P_RING_SOURCE);
    const float* ring_activation = leo_p_cptr<float>(p, LEO_P_RING_ACTIVATION);
    const float* ring_weight = leo_p_cptr<float>(p, LEO_P_RING_WEIGHT);
    float* branch_delta = leo_p_ptr<float>(p, LEO_P_BRANCH_DELTA);
    float* rec_bs = leo_p_ptr<float>(p, LEO_P_REC_BRANCH_SENSITIVITY);
    float* rec_ms = leo_p_ptr<float>(p, LEO_P_REC_MEMBRANE_SENSITIVITY);
    float* rec_fs = leo_p_ptr<float>(p, LEO_P_REC_FATIGUE_SENSITIVITY);
    float* rec_af = leo_p_ptr<float>(p, LEO_P_REC_ADAPTATION_FAST_SENSITIVITY);
    float* rec_am = leo_p_ptr<float>(p, LEO_P_REC_ADAPTATION_MEDIUM_SENSITIVITY);
    float* rec_as = leo_p_ptr<float>(p, LEO_P_REC_ADAPTATION_SLOW_SENSITIVITY);
    unsigned long long* rec_last = leo_p_ptr<unsigned long long>(p, LEO_P_REC_LAST_TICK);
    unsigned int* rec_mark = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_MARK);
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();
    // The per-step fallback launches for max_active_global records at every delay
    // and immediately rejects records beyond ring_count[bucket]. The persistent
    // kernel can read ring_count on-device and iterate only records that exist.
    for (unsigned int delay_index = 0U; delay_index < 4U; ++delay_index) {
        const unsigned int delay_value = delay_index == 0U ? 1U : (delay_index == 1U ? 2U : (delay_index == 2U ? 4U : 8U));
        const unsigned int bucket = (unsigned int)((tick + 9ULL - (unsigned long long)delay_value) % 9ULL);
        const unsigned int count = ring_count[bucket];
        const unsigned int total = count * cfg->synapses_per_neuron;
        for (unsigned int inner = thread; inner < total; inner += stride) {
            const unsigned int record = inner / cfg->synapses_per_neuron;
            const unsigned int local_slot = inner - record * cfg->synapses_per_neuron;
            const unsigned int source_position = bucket * cfg->max_active_global + record;
            const unsigned int source = ring_source[source_position];
            const unsigned int slot = source * cfg->synapses_per_neuron + local_slot;
            if ((unsigned int)recurrent_delay[slot] != delay_value) continue;
            const float presynaptic = ring_activation[source_position];
            if (learning_trace) {
                leo_advance_recurrent_trace(
                    slot, tick, cfg, rec_bs, rec_ms, rec_fs, rec_af, rec_am, rec_as, rec_last
                );
                rec_bs[slot] += presynaptic;
                if (atomicCAS(&rec_mark[slot], 0U, 1U) == 0U) {
                    const unsigned int position = atomicAdd(rec_count, 1U);
                    rec_list[position] = slot;
                }
            }
            const unsigned int target = recurrent_target[slot];
            const unsigned int branch = (unsigned int)recurrent_branch[slot];
            const unsigned long long snapshot =
                ((unsigned long long)source_position * cfg->synapses_per_neuron) + local_slot;
            atomicAdd(&branch_delta[(unsigned long long)target * 4ULL + branch], presynaptic * ring_weight[snapshot]);
            leo_p_mark_touched(p, target, tick + 1ULL);
        }
    }
}

__device__ void leo_p_inject_symbol(
    const unsigned long long* p,
    unsigned long long tick,
    unsigned int symbol,
    bool learning_trace,
    unsigned int* eligible_list,
    unsigned int* eligible_count
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* input_target = leo_p_cptr<unsigned int>(p, LEO_P_INPUT_TARGET);
    const unsigned char* input_branch = leo_p_cptr<unsigned char>(p, LEO_P_INPUT_BRANCH);
    const float* input_weight = leo_p_cptr<float>(p, LEO_P_INPUT_WEIGHT);
    float* branch_delta = leo_p_ptr<float>(p, LEO_P_BRANCH_DELTA);
    float* bs = leo_p_ptr<float>(p, LEO_P_INPUT_BRANCH_SENSITIVITY);
    float* ms = leo_p_ptr<float>(p, LEO_P_INPUT_MEMBRANE_SENSITIVITY);
    float* fs = leo_p_ptr<float>(p, LEO_P_INPUT_FATIGUE_SENSITIVITY);
    float* af = leo_p_ptr<float>(p, LEO_P_INPUT_ADAPTATION_FAST_SENSITIVITY);
    float* am = leo_p_ptr<float>(p, LEO_P_INPUT_ADAPTATION_MEDIUM_SENSITIVITY);
    float* as = leo_p_ptr<float>(p, LEO_P_INPUT_ADAPTATION_SLOW_SENSITIVITY);
    unsigned long long* last_tick = leo_p_ptr<unsigned long long>(p, LEO_P_INPUT_LAST_TICK);
    unsigned int* mark = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_MARK);
    unsigned int* list = eligible_list;
    unsigned int* count = eligible_count;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();
    for (unsigned int local = thread; local < cfg->input_fanout; local += stride) {
        const unsigned int slot = symbol * cfg->input_fanout + local;
        if (learning_trace) {
            leo_advance_recurrent_trace(slot, tick, cfg, bs, ms, fs, af, am, as, last_tick);
            bs[slot] += 1.0f;
            if (atomicCAS(&mark[slot], 0U, 1U) == 0U) {
                const unsigned int position = atomicAdd(count, 1U);
                list[position] = slot;
            }
        }
        const unsigned int target = input_target[slot];
        const unsigned int branch = (unsigned int)input_branch[slot];
        atomicAdd(&branch_delta[(unsigned long long)target * 4ULL + branch], input_weight[slot]);
        leo_p_mark_touched(p, target, tick + 1ULL);
    }
}

// Frozen replay-prefix advancement only needs context history for the
// next supervised step. Active context lookup/projection is output-only work
// and is recomputed by the first subsequent ordinary step.
__device__ void leo_p_context_advance_history(
    const unsigned long long* p,
    unsigned int symbol
) {
    if (leo_p_global_thread() != 0U) return;
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    unsigned int* history = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_HISTORY);
    unsigned int* history_count = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_HISTORY_COUNT);
    unsigned int count = *history_count;
    if (count == cfg->context_max_order) {
        for (unsigned int index = 1U; index < count; ++index) {
            history[index - 1U] = history[index];
        }
        count -= 1U;
    }
    history[count++] = symbol;
    *history_count = count;
}

__device__ void leo_p_context_resolve(
    const unsigned long long* p,
    unsigned int symbol,
    bool enabled,
    bool allocate
) {
    if (leo_p_global_thread() != 0U) return;
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    unsigned int* history = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_HISTORY);
    unsigned int* history_count = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_HISTORY_COUNT);
    unsigned long long* context_keys = leo_p_ptr<unsigned long long>(p, LEO_P_CONTEXT_KEYS);
    float* context_embeddings = leo_p_ptr<float>(p, LEO_P_CONTEXT_EMBEDDINGS);
    unsigned int* context_observations = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_OBSERVATIONS);
    unsigned int* active_context_slots = leo_p_ptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_SLOTS);
    float* active_context_scales = leo_p_ptr<float>(p, LEO_P_ACTIVE_CONTEXT_SCALES);
    unsigned int* active_context_count = leo_p_ptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_COUNT);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    unsigned int count = *history_count;
    if (count == cfg->context_max_order) {
        for (unsigned int index = 1U; index < count; ++index) history[index - 1U] = history[index];
        count -= 1U;
    }
    history[count++] = symbol;
    *history_count = count;
    *active_context_count = 0U;
    if (!enabled) return;
    const unsigned int available = count < cfg->context_max_order ? count : cfg->context_max_order;
    float scale_sum = 0.0f;
    unsigned int resolved_count = 0U;
    unsigned int probe_sum = 0U;
    const float confidence_target = (float)cfg->context_confidence_observations;
    for (unsigned int order = 1U; order <= available; ++order) {
        const unsigned long long key = leo_context_key(history, count, order);
        const unsigned int base = (order - 1U) * cfg->context_slots_per_order;
        const unsigned int start = (unsigned int)(key % (unsigned long long)cfg->context_slots_per_order);
        unsigned int empty_slot = 0xffffffffU;
        unsigned int weakest_slot = base + start;
        unsigned int weakest_observations = 0xffffffffU;
        unsigned int found_slot = 0xffffffffU;
        for (unsigned int offset = 0U; offset < cfg->context_probe_limit; ++offset) {
            const unsigned int slot = base + (start + offset) % cfg->context_slots_per_order;
            probe_sum += 1U;
            const unsigned long long existing = context_keys[slot];
            if (existing == key) {
                found_slot = slot;
                break;
            }
            if (existing == 0ULL && empty_slot == 0xffffffffU) empty_slot = slot;
            const unsigned int observations = context_observations[slot];
            if (observations < weakest_observations) {
                weakest_observations = observations;
                weakest_slot = slot;
            }
        }
        if (found_slot == 0xffffffffU) {
            if (!allocate) continue;
            found_slot = empty_slot != 0xffffffffU ? empty_slot : weakest_slot;
            if (context_keys[found_slot] != key) {
                context_keys[found_slot] = key;
                context_observations[found_slot] = 0U;
                const unsigned long long row = (unsigned long long)found_slot * cfg->context_embedding_dim;
                for (unsigned int dimension = 0U; dimension < cfg->context_embedding_dim; ++dimension) {
                    context_embeddings[row + dimension] = 0.0f;
                }
                leo_mark_changed(found_slot, changed_marks, changed_list, changed_count);
            }
        }
        const float observations = (float)context_observations[found_slot];
        const float confidence = observations / (observations + confidence_target);
        const float scale = confidence * (float)order;
        active_context_slots[resolved_count] = found_slot;
        active_context_scales[resolved_count] = scale;
        scale_sum += scale;
        resolved_count += 1U;
    }
    if (scale_sum > 0.0f) {
        for (unsigned int index = 0U; index < resolved_count; ++index) {
            active_context_scales[index] /= scale_sum;
        }
    }
    *active_context_count = resolved_count;
    counters->context_cells = resolved_count;
    counters->context_probes = probe_sum;
}

// Shared-model context resolution for the Stage 3 wavefront path. Context
// keys and parameters are canonical model state, so multiple stories may probe
// them concurrently. Empty slots are claimed with atomicCAS. Unlike the per-step fallback
// single-story resolver, this path deliberately does not evict the weakest
// occupied slot during a concurrent wavefront; that avoids one lane replacing
// a context while another lane is reading it.
__device__ void leo_p_context_resolve_shared(
    const unsigned long long* p,
    unsigned int symbol,
    bool enabled,
    bool allocate
) {
    if (leo_p_global_thread() != 0U) return;
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    unsigned int* history = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_HISTORY);
    unsigned int* history_count = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_HISTORY_COUNT);
    unsigned long long* context_keys = leo_p_ptr<unsigned long long>(p, LEO_P_CONTEXT_KEYS);
    const unsigned int* context_observations = leo_p_cptr<unsigned int>(p, LEO_P_CONTEXT_OBSERVATIONS);
    unsigned int* active_context_slots = leo_p_ptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_SLOTS);
    float* active_context_scales = leo_p_ptr<float>(p, LEO_P_ACTIVE_CONTEXT_SCALES);
    unsigned int* active_context_count = leo_p_ptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_COUNT);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);

    unsigned int count = *history_count;
    if (count == cfg->context_max_order) {
        for (unsigned int index = 1U; index < count; ++index) history[index - 1U] = history[index];
        count -= 1U;
    }
    history[count++] = symbol;
    *history_count = count;
    *active_context_count = 0U;
    if (!enabled) return;

    const unsigned int available = count < cfg->context_max_order ? count : cfg->context_max_order;
    const float confidence_target = (float)cfg->context_confidence_observations;
    float scale_sum = 0.0f;
    unsigned int resolved_count = 0U;
    unsigned int probe_sum = 0U;

    for (unsigned int order = 1U; order <= available; ++order) {
        const unsigned long long key = leo_context_key(history, count, order);
        const unsigned int base = (order - 1U) * cfg->context_slots_per_order;
        const unsigned int start = (unsigned int)(key % (unsigned long long)cfg->context_slots_per_order);
        unsigned int found_slot = 0xffffffffU;

        for (unsigned int offset = 0U; offset < cfg->context_probe_limit; ++offset) {
            const unsigned int slot = base + (start + offset) % cfg->context_slots_per_order;
            probe_sum += 1U;
            const unsigned long long existing = context_keys[slot];
            if (existing == key) {
                found_slot = slot;
                break;
            }
            if (allocate && existing == 0ULL) {
                const unsigned long long previous = atomicCAS(
                    reinterpret_cast<unsigned long long*>(&context_keys[slot]),
                    0ULL,
                    key
                );
                if (previous == 0ULL || previous == key) {
                    found_slot = slot;
                    if (previous == 0ULL) {
                        leo_mark_changed(slot, changed_marks, changed_list, changed_count);
                    }
                    break;
                }
            }
        }

        if (found_slot == 0xffffffffU) continue;
        const float observations = (float)context_observations[found_slot];
        const float confidence = observations / (observations + confidence_target);
        const float scale = confidence * (float)order;
        active_context_slots[resolved_count] = found_slot;
        active_context_scales[resolved_count] = scale;
        scale_sum += scale;
        resolved_count += 1U;
    }

    if (scale_sum > 0.0f) {
        for (unsigned int index = 0U; index < resolved_count; ++index) {
            active_context_scales[index] /= scale_sum;
        }
    }
    *active_context_count = resolved_count;
    counters->context_cells = resolved_count;
    counters->context_probes = probe_sum;
}

__device__ __forceinline__ void leo_p_select_model_block(
    const unsigned long long* p,
    unsigned long long tick,
    unsigned int model_block
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    if (model_block >= cfg->block_count) return;
    const unsigned long long tag = tick + 1ULL;
    const float* threshold = leo_p_cptr<float>(p, LEO_P_THRESHOLD);
    const float* excitability = leo_p_cptr<float>(p, LEO_P_EXCITABILITY);
    float* membrane = leo_p_ptr<float>(p, LEO_P_MEMBRANE);
    float* fatigue = leo_p_ptr<float>(p, LEO_P_FATIGUE);
    unsigned long long* refractory_until = leo_p_ptr<unsigned long long>(p, LEO_P_REFRACTORY_UNTIL);
    float* branches = leo_p_ptr<float>(p, LEO_P_BRANCHES);
    float* branch_delta = leo_p_ptr<float>(p, LEO_P_BRANCH_DELTA);
    unsigned long long* branch_last_tick = leo_p_ptr<unsigned long long>(p, LEO_P_BRANCH_LAST_TICK);
    unsigned long long* neuron_last_tick = leo_p_ptr<unsigned long long>(p, LEO_P_NEURON_LAST_TICK);
    float* adaptation_fast = leo_p_ptr<float>(p, LEO_P_ADAPTATION_FAST);
    float* adaptation_medium = leo_p_ptr<float>(p, LEO_P_ADAPTATION_MEDIUM);
    float* adaptation_slow = leo_p_ptr<float>(p, LEO_P_ADAPTATION_SLOW);
    const unsigned long long* touched_epoch = leo_p_cptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    const float* population_inhibition = leo_p_cptr<float>(p, LEO_P_POPULATION_INHIBITION);
    float* candidate_activation = leo_p_ptr<float>(p, LEO_P_CANDIDATE_ACTIVATION);
    unsigned int* winner_neuron = leo_p_ptr<unsigned int>(p, LEO_P_BLOCK_WINNER_NEURON);
    float* winner_value = leo_p_ptr<float>(p, LEO_P_BLOCK_WINNER_VALUE);
    float* block_cutoff = leo_p_ptr<float>(p, LEO_P_BLOCK_CUTOFF);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);

    for (unsigned int local = threadIdx.x; local < cfg->neurons_per_block; local += blockDim.x) {
        const unsigned int neuron = model_block * cfg->neurons_per_block + local;
        float candidate = 0.0f;
        if (touched_epoch[neuron] == tag) {
            const unsigned long long branch_elapsed = tick >= branch_last_tick[neuron]
                ? tick - branch_last_tick[neuron] : 0ULL;
            const unsigned long long base = (unsigned long long)neuron * 4ULL;
            if (branch_elapsed > 0ULL) {
                const float decay = leo_decay(cfg->branch_decay, branch_elapsed);
                for (unsigned int branch = 0U; branch < 4U; ++branch) branches[base + branch] *= decay;
                branch_last_tick[neuron] = tick;
            }
            for (unsigned int branch = 0U; branch < 4U; ++branch) {
                branches[base + branch] += branch_delta[base + branch];
                branch_delta[base + branch] = 0.0f;
            }
            const unsigned long long neuron_elapsed = tick >= neuron_last_tick[neuron]
                ? tick - neuron_last_tick[neuron] : 0ULL;
            if (neuron_elapsed > 0ULL) {
                membrane[neuron] *= leo_decay(cfg->membrane_decay, neuron_elapsed);
                fatigue[neuron] *= leo_decay(cfg->fatigue_decay, neuron_elapsed);
                adaptation_fast[neuron] *= leo_decay(cfg->adaptation_fast_decay, neuron_elapsed);
                adaptation_medium[neuron] *= leo_decay(cfg->adaptation_medium_decay, neuron_elapsed);
                adaptation_slow[neuron] *= leo_decay(cfg->adaptation_slow_decay, neuron_elapsed);
                neuron_last_tick[neuron] = tick;
            }
            if (refractory_until[neuron] <= tick) {
                const float gate = leo_clamp(branches[base + 3ULL], 0.0f, 1.0f);
                const float evidence = branches[base] + branches[base + 2ULL] * gate + branches[base + 1ULL];
                const float adaptation = adaptation_fast[neuron] + adaptation_medium[neuron] + adaptation_slow[neuron];
                const float drive = excitability[neuron] * evidence - fatigue[neuron] - adaptation - *population_inhibition;
                membrane[neuron] += drive;
                candidate = leo_clamp(membrane[neuron] - threshold[neuron], 0.0f, 1.0f);
                if (candidate > 0.0f) atomicAdd(&counters->suprathreshold, 1U);
            }
        }
        candidate_activation[neuron] = candidate;
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        const unsigned int rotation = (unsigned int)(tick % (unsigned long long)cfg->neuron_count);
        const unsigned int keep = cfg->max_active_per_block;
        float top_value[LEO_MAX_BLOCK_WINNERS + 1];
        unsigned int top_neuron[LEO_MAX_BLOCK_WINNERS + 1];
        for (unsigned int i = 0U; i <= keep; ++i) {
            top_value[i] = -1.0f;
            top_neuron[i] = 0U;
        }
        unsigned int positive = 0U;
        const unsigned int start = model_block * cfg->neurons_per_block;
        for (unsigned int local = 0U; local < cfg->neurons_per_block; ++local) {
            const unsigned int candidate_neuron = start + local;
            const float value = candidate_activation[candidate_neuron];
            if (value <= 0.0f) continue;
            positive += 1U;
            unsigned int position = keep;
            for (unsigned int scan = 0U; scan <= keep; ++scan) {
                if (top_value[scan] < 0.0f || leo_better(
                    value,
                    candidate_neuron,
                    top_value[scan],
                    top_neuron[scan],
                    rotation,
                    cfg->neuron_count
                )) {
                    position = scan;
                    break;
                }
            }
            if (position <= keep) {
                for (unsigned int move = keep; move > position; --move) {
                    top_value[move] = top_value[move - 1U];
                    top_neuron[move] = top_neuron[move - 1U];
                }
                top_value[position] = value;
                top_neuron[position] = candidate_neuron;
            }
        }
        const unsigned int winner_count = positive < keep ? positive : keep;
        const unsigned int winner_base = model_block * keep;
        for (unsigned int i = 0U; i < keep; ++i) {
            if (i < winner_count) {
                winner_neuron[winner_base + i] = top_neuron[i];
                winner_value[winner_base + i] = top_value[i];
            } else {
                winner_neuron[winner_base + i] = 0U;
                winner_value[winner_base + i] = -1.0f;
            }
        }
        block_cutoff[model_block] = positive > keep ? top_value[keep] : 0.0f;
        atomicAdd(&counters->block_selected, winner_count);
    }
}

__device__ void leo_p_select_blocks(const unsigned long long* p, unsigned long long tick) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    for (unsigned int model_block = 0U; model_block < cfg->block_count; ++model_block) {
        leo_p_select_model_block(p, tick, model_block);
        __syncthreads();
    }
}

__device__ void leo_p_select_global(
    const unsigned long long* p,
    unsigned long long tick,
    float* shared_values,
    unsigned int* shared_neurons
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* block_winner_neuron = leo_p_cptr<unsigned int>(p, LEO_P_BLOCK_WINNER_NEURON);
    const float* block_winner_value = leo_p_cptr<float>(p, LEO_P_BLOCK_WINNER_VALUE);
    unsigned int* active = leo_p_ptr<unsigned int>(p, LEO_P_ACTIVE);
    float* active_value = leo_p_ptr<float>(p, LEO_P_ACTIVE_VALUE);
    unsigned int* active_count = leo_p_ptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    float* activation = leo_p_ptr<float>(p, LEO_P_ACTIVATION);
    unsigned long long* selected_epoch = leo_p_ptr<unsigned long long>(p, LEO_P_SELECTED_EPOCH);
    float* population_cutoff = leo_p_ptr<float>(p, LEO_P_POPULATION_CUTOFF);
    float* population_inhibition = leo_p_ptr<float>(p, LEO_P_POPULATION_INHIBITION);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    const unsigned int lane = threadIdx.x;
    const unsigned int available = cfg->block_count * cfg->max_active_per_block;
    for (unsigned int index = lane; index < LEO_GLOBAL_SORT; index += blockDim.x) {
        if (index < available) {
            shared_values[index] = block_winner_value[index];
            shared_neurons[index] = block_winner_neuron[index];
        } else {
            shared_values[index] = -1.0f;
            shared_neurons[index] = 0U;
        }
    }
    __syncthreads();
    const unsigned int rotation = (unsigned int)(tick % (unsigned long long)cfg->neuron_count);
    for (unsigned int width = 2U; width <= LEO_GLOBAL_SORT; width <<= 1U) {
        for (unsigned int stride = width >> 1U; stride > 0U; stride >>= 1U) {
            for (unsigned int index = lane; index < LEO_GLOBAL_SORT; index += blockDim.x) {
                const unsigned int other = index ^ stride;
                if (other > index) {
                    const bool better_first = (index & width) == 0U;
                    const bool left_better = leo_better(
                        shared_values[index], shared_neurons[index], shared_values[other], shared_neurons[other],
                        rotation, cfg->neuron_count
                    );
                    const bool should_swap = better_first ? !left_better : left_better;
                    if (should_swap) {
                        const float value = shared_values[index];
                        shared_values[index] = shared_values[other];
                        shared_values[other] = value;
                        const unsigned int neuron = shared_neurons[index];
                        shared_neurons[index] = shared_neurons[other];
                        shared_neurons[other] = neuron;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (lane == 0U) {
        unsigned int selected = 0U;
        while (selected < available && selected < LEO_GLOBAL_SORT && shared_values[selected] > 0.0f) selected += 1U;
        const unsigned int total = selected;
        const unsigned int cap = cfg->max_active_global;
        const unsigned int kept = total < cap ? total : cap;
        *active_count = kept;
        *population_cutoff = total > cap ? leo_clamp(shared_values[cap], 0.0f, cfg->population_inhibition_max) : 0.0f;
        counters->global_clipped = total > cap ? total - cap : 0U;
        const float activity = (float)kept / (float)(cfg->neuron_count == 0U ? 1U : cfg->neuron_count);
        const float error = activity - cfg->target_activity;
        *population_inhibition = leo_clamp(
            *population_inhibition + cfg->population_inhibition_rate * error,
            0.0f, cfg->population_inhibition_max
        );
        for (unsigned int index = 0U; index < kept; ++index) {
            const unsigned int neuron = shared_neurons[index];
            active[index] = neuron;
            active_value[index] = shared_values[index];
            activation[neuron] = shared_values[index];
            selected_epoch[neuron] = tick + 1ULL;
            leo_p_mark_learning_destination(p, neuron, tick + 1ULL);
        }
    }
}

__device__ void leo_p_cache_surrogate_work(const unsigned long long* p, unsigned long long tick,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const float* threshold = leo_p_cptr<float>(p, LEO_P_THRESHOLD);
    const float* membrane = leo_p_cptr<float>(p, LEO_P_MEMBRANE);
    const unsigned long long* refractory = leo_p_cptr<unsigned long long>(p, LEO_P_REFRACTORY_UNTIL);
    const unsigned long long* touched = leo_p_cptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    const unsigned long long* selected = leo_p_cptr<unsigned long long>(p, LEO_P_SELECTED_EPOCH);
    const unsigned int* touched_list = leo_p_cptr<unsigned int>(p, LEO_P_TOUCHED_LIST);
    const unsigned int* touched_count = leo_p_cptr<unsigned int>(p, LEO_P_TOUCHED_COUNT);
    const float* block_cutoff = leo_p_cptr<float>(p, LEO_P_BLOCK_CUTOFF);
    const float* population_cutoff = leo_p_cptr<float>(p, LEO_P_POPULATION_CUTOFF);
    float* surrogate = leo_p_ptr<float>(p, LEO_P_SURROGATE);
    const unsigned long long tag = tick + 1ULL;
    const unsigned int count = *touched_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int neuron = touched_list[position];
        if (neuron >= cfg->neuron_count || touched[neuron] != tag || refractory[neuron] > tick) {
            continue;
        }
        const float margin = membrane[neuron] - threshold[neuron];
        const unsigned int block = neuron / cfg->neurons_per_block;
        const float cutoff = block_cutoff[block] > *population_cutoff ? block_cutoff[block] : *population_cutoff;
        surrogate[neuron] = leo_surrogate(
            margin, cutoff, selected[neuron] == tag, cfg->surrogate_width, cfg->surrogate_gain
        );
    }

}

__device__ void leo_p_cache_surrogate_worklist(const unsigned long long* p, unsigned long long tick) {
    leo_p_cache_surrogate_work(
        p, tick, leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_cache_surrogate(const unsigned long long* p, unsigned long long tick) {
    leo_p_cache_surrogate_worklist(p, tick);
}

__device__ void leo_p_update_recurrent_eligibility_work(
    const unsigned long long* p,
    unsigned long long tick,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    unsigned int* next_list,
    unsigned int* next_count,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* target = leo_p_cptr<unsigned int>(p, LEO_P_RECURRENT_TARGET);
    const unsigned char* branch = leo_p_cptr<unsigned char>(p, LEO_P_RECURRENT_BRANCH);
    const float* excitability = leo_p_cptr<float>(p, LEO_P_EXCITABILITY);
    const float* branches = leo_p_cptr<float>(p, LEO_P_BRANCHES);
    const unsigned long long* refractory = leo_p_cptr<unsigned long long>(p, LEO_P_REFRACTORY_UNTIL);
    const unsigned long long* touched = leo_p_cptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    const unsigned long long* selected = leo_p_cptr<unsigned long long>(p, LEO_P_SELECTED_EPOCH);
    const float* surrogate = leo_p_cptr<float>(p, LEO_P_SURROGATE);
    float* bs = leo_p_ptr<float>(p, LEO_P_REC_BRANCH_SENSITIVITY);
    float* ms = leo_p_ptr<float>(p, LEO_P_REC_MEMBRANE_SENSITIVITY);
    float* fs = leo_p_ptr<float>(p, LEO_P_REC_FATIGUE_SENSITIVITY);
    float* af = leo_p_ptr<float>(p, LEO_P_REC_ADAPTATION_FAST_SENSITIVITY);
    float* am = leo_p_ptr<float>(p, LEO_P_REC_ADAPTATION_MEDIUM_SENSITIVITY);
    float* as = leo_p_ptr<float>(p, LEO_P_REC_ADAPTATION_SLOW_SENSITIVITY);
    float* eligibility = leo_p_ptr<float>(p, LEO_P_REC_ELIGIBILITY);
    unsigned long long* last_tick = leo_p_ptr<unsigned long long>(p, LEO_P_REC_LAST_TICK);
    unsigned int* mark = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_MARK);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        if (mark[slot] == 0U) continue;
        leo_advance_recurrent_trace(slot, tick, cfg, bs, ms, fs, af, am, as, last_tick);
        const unsigned int destination = target[slot];
        const unsigned long long tag = tick + 1ULL;
        float e;
        if (touched[destination] != tag || refractory[destination] > tick) {
            e = eligibility[slot] * cfg->eligibility_decay;
        } else {
            const unsigned long long base = (unsigned long long)destination * 4ULL;
            const float jacobian = leo_branch_jacobian(
                branch[slot], excitability[destination], branches[base + 2ULL], branches[base + 3ULL]
            );
            const float adaptation_sum = af[slot] + am[slot] + as[slot];
            const float membrane_value = ms[slot] + jacobian * bs[slot] - fs[slot] - adaptation_sum;
            ms[slot] = membrane_value;
            e = surrogate[destination] * membrane_value;
        }
        eligibility[slot] = e;
        if (touched[destination] == tag && selected[destination] == tag) {
            fs[slot] += cfg->fatigue_gain * e;
            af[slot] += cfg->adaptation_fast_gain * e;
            am[slot] += cfg->adaptation_medium_gain * e;
            as[slot] += cfg->adaptation_slow_gain * e;
        }
        float magnitude = fabsf(bs[slot]);
        magnitude = fmaxf(magnitude, fabsf(ms[slot]));
        magnitude = fmaxf(magnitude, fabsf(fs[slot]));
        magnitude = fmaxf(magnitude, fabsf(af[slot]));
        magnitude = fmaxf(magnitude, fabsf(am[slot]));
        magnitude = fmaxf(magnitude, fabsf(as[slot]));
        magnitude = fmaxf(magnitude, fabsf(e));
        if (magnitude > cfg->eligibility_epsilon) {
            const unsigned int next = atomicAdd(next_count, 1U);
            next_list[next] = slot;
            atomicAdd(&counters->eligible_recurrent, 1U);
            leo_p_mark_learning_destination(p, destination, tag);
        } else {
            mark[slot] = 0U;
        }
    }

}

__device__ void leo_p_update_recurrent_eligibility(
    const unsigned long long* p,
    unsigned long long tick,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    unsigned int* next_list,
    unsigned int* next_count
) {
    leo_p_update_recurrent_eligibility_work(
        p, tick, eligible_list, eligible_count, next_list, next_count,
        leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_update_input_eligibility_work(
    const unsigned long long* p,
    unsigned long long tick,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    unsigned int* next_list,
    unsigned int* next_count,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* target = leo_p_cptr<unsigned int>(p, LEO_P_INPUT_TARGET);
    const unsigned char* branch = leo_p_cptr<unsigned char>(p, LEO_P_INPUT_BRANCH);
    const float* excitability = leo_p_cptr<float>(p, LEO_P_EXCITABILITY);
    const float* branches = leo_p_cptr<float>(p, LEO_P_BRANCHES);
    const unsigned long long* refractory = leo_p_cptr<unsigned long long>(p, LEO_P_REFRACTORY_UNTIL);
    const unsigned long long* touched = leo_p_cptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    const unsigned long long* selected = leo_p_cptr<unsigned long long>(p, LEO_P_SELECTED_EPOCH);
    const float* surrogate = leo_p_cptr<float>(p, LEO_P_SURROGATE);
    float* bs = leo_p_ptr<float>(p, LEO_P_INPUT_BRANCH_SENSITIVITY);
    float* ms = leo_p_ptr<float>(p, LEO_P_INPUT_MEMBRANE_SENSITIVITY);
    float* fs = leo_p_ptr<float>(p, LEO_P_INPUT_FATIGUE_SENSITIVITY);
    float* af = leo_p_ptr<float>(p, LEO_P_INPUT_ADAPTATION_FAST_SENSITIVITY);
    float* am = leo_p_ptr<float>(p, LEO_P_INPUT_ADAPTATION_MEDIUM_SENSITIVITY);
    float* as = leo_p_ptr<float>(p, LEO_P_INPUT_ADAPTATION_SLOW_SENSITIVITY);
    float* eligibility = leo_p_ptr<float>(p, LEO_P_INPUT_ELIGIBILITY);
    unsigned long long* last_tick = leo_p_ptr<unsigned long long>(p, LEO_P_INPUT_LAST_TICK);
    unsigned int* mark = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_MARK);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        if (mark[slot] == 0U) continue;
        leo_advance_recurrent_trace(slot, tick, cfg, bs, ms, fs, af, am, as, last_tick);
        const unsigned int destination = target[slot];
        const unsigned long long tag = tick + 1ULL;
        float e;
        if (touched[destination] != tag || refractory[destination] > tick) {
            e = eligibility[slot] * cfg->eligibility_decay;
        } else {
            const unsigned long long base = (unsigned long long)destination * 4ULL;
            const float jacobian = leo_branch_jacobian(
                branch[slot], excitability[destination], branches[base + 2ULL], branches[base + 3ULL]
            );
            const float adaptation_sum = af[slot] + am[slot] + as[slot];
            const float membrane_value = ms[slot] + jacobian * bs[slot] - fs[slot] - adaptation_sum;
            ms[slot] = membrane_value;
            e = surrogate[destination] * membrane_value;
        }
        eligibility[slot] = e;
        if (touched[destination] == tag && selected[destination] == tag) {
            fs[slot] += cfg->fatigue_gain * e;
            af[slot] += cfg->adaptation_fast_gain * e;
            am[slot] += cfg->adaptation_medium_gain * e;
            as[slot] += cfg->adaptation_slow_gain * e;
        }
        float magnitude = fabsf(bs[slot]);
        magnitude = fmaxf(magnitude, fabsf(ms[slot]));
        magnitude = fmaxf(magnitude, fabsf(fs[slot]));
        magnitude = fmaxf(magnitude, fabsf(af[slot]));
        magnitude = fmaxf(magnitude, fabsf(am[slot]));
        magnitude = fmaxf(magnitude, fabsf(as[slot]));
        magnitude = fmaxf(magnitude, fabsf(e));
        if (magnitude > cfg->eligibility_epsilon) {
            const unsigned int next = atomicAdd(next_count, 1U);
            next_list[next] = slot;
            atomicAdd(&counters->eligible_input, 1U);
            leo_p_mark_learning_destination(p, destination, tag);
        } else {
            mark[slot] = 0U;
        }
    }

}

__device__ void leo_p_update_input_eligibility(
    const unsigned long long* p,
    unsigned long long tick,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    unsigned int* next_list,
    unsigned int* next_count
) {
    leo_p_update_input_eligibility_work(
        p, tick, eligible_list, eligible_count, next_list, next_count,
        leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ __forceinline__ void leo_p_post_and_emit_work(
    const unsigned long long* p,
    unsigned long long tick,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const float* threshold = leo_p_cptr<float>(p, LEO_P_THRESHOLD);
    const float* recurrent_weight = leo_p_cptr<float>(p, LEO_P_RECURRENT_WEIGHT);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const float* active_value = leo_p_cptr<float>(p, LEO_P_ACTIVE_VALUE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    float* membrane = leo_p_ptr<float>(p, LEO_P_MEMBRANE);
    float* fatigue = leo_p_ptr<float>(p, LEO_P_FATIGUE);
    unsigned long long* refractory = leo_p_ptr<unsigned long long>(p, LEO_P_REFRACTORY_UNTIL);
    float* af = leo_p_ptr<float>(p, LEO_P_ADAPTATION_FAST);
    float* am = leo_p_ptr<float>(p, LEO_P_ADAPTATION_MEDIUM);
    float* as = leo_p_ptr<float>(p, LEO_P_ADAPTATION_SLOW);
    unsigned int* ring_count = leo_p_ptr<unsigned int>(p, LEO_P_RING_COUNT);
    unsigned int* ring_source = leo_p_ptr<unsigned int>(p, LEO_P_RING_SOURCE);
    float* ring_activation = leo_p_ptr<float>(p, LEO_P_RING_ACTIVATION);
    float* ring_weight = leo_p_ptr<float>(p, LEO_P_RING_WEIGHT);
    const unsigned int count = *active_count;
    const unsigned int bucket = (unsigned int)(tick % 9ULL);
    if (thread == 0U) ring_count[bucket] = count;
    for (unsigned int index = thread; index < count; index += stride) {
        const unsigned int neuron = active[index];
        const float value = active_value[index];
        fatigue[neuron] += cfg->fatigue_gain * value;
        af[neuron] += cfg->adaptation_fast_gain * value;
        am[neuron] += cfg->adaptation_medium_gain * value;
        as[neuron] += cfg->adaptation_slow_gain * value;
        membrane[neuron] -= threshold[neuron] * cfg->membrane_reset_fraction;
        refractory[neuron] = tick + (unsigned long long)cfg->refractory_ticks + 1ULL;
        const unsigned int ring_position = bucket * cfg->max_active_global + index;
        ring_source[ring_position] = neuron;
        ring_activation[ring_position] = value;
        const unsigned long long ring_base = (unsigned long long)ring_position * cfg->synapses_per_neuron;
        const unsigned long long source_base = (unsigned long long)neuron * cfg->synapses_per_neuron;
        for (unsigned int local = 0U; local < cfg->synapses_per_neuron; ++local) {
            ring_weight[ring_base + local] = recurrent_weight[source_base + local];
        }
    }
}

__device__ void leo_p_post_and_emit(const unsigned long long* p, unsigned long long tick) {
    leo_p_post_and_emit_work(p, tick, leo_p_global_thread(), leo_p_global_stride());
}

__device__ __forceinline__ void leo_p_post_and_emit_grid(
    const unsigned long long* p,
    unsigned long long tick
) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    leo_p_post_and_emit_work(p, tick, thread, stride);
}

__device__ void leo_p_forward(
    const unsigned long long* p,
    int target_index,
    float* latent,
    float* reduction
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const float* output_weights = leo_p_cptr<float>(p, LEO_P_OUTPUT_WEIGHT);
    const float* output_bias = leo_p_cptr<float>(p, LEO_P_OUTPUT_BIAS);
    const float* context_embeddings = leo_p_cptr<float>(p, LEO_P_CONTEXT_EMBEDDINGS);
    const float* context_output_weights = leo_p_cptr<float>(p, LEO_P_CONTEXT_OUTPUT_WEIGHT);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const float* active_value = leo_p_cptr<float>(p, LEO_P_ACTIVE_VALUE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const unsigned int* context_slots = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_SLOTS);
    const float* context_scales = leo_p_cptr<float>(p, LEO_P_ACTIVE_CONTEXT_SCALES);
    const unsigned int* context_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_COUNT);
    float* neural_logits = leo_p_ptr<float>(p, LEO_P_NEURAL_LOGITS);
    float* logits = leo_p_ptr<float>(p, LEO_P_LOGITS);
    float* probabilities = leo_p_ptr<float>(p, LEO_P_PROBABILITIES);
    float* errors = leo_p_ptr<float>(p, LEO_P_ERRORS);
    float* context_latent = leo_p_ptr<float>(p, LEO_P_CONTEXT_LATENT);
    float* context_gradient = leo_p_ptr<float>(p, LEO_P_CONTEXT_GRADIENT);
    const unsigned int lane = threadIdx.x;
    const unsigned int embedding_dim = cfg->context_embedding_dim;
    const unsigned int contexts = *context_count;
    if (lane < embedding_dim) {
        float value = 0.0f;
        for (unsigned int index = 0U; index < contexts; ++index) {
            const unsigned int slot = context_slots[index];
            value += context_embeddings[(unsigned long long)slot * embedding_dim + lane] * context_scales[index];
        }
        latent[lane] = value;
        context_latent[lane] = value;
    }
    __syncthreads();
    for (unsigned int output = lane; output < 512U; output += blockDim.x) {
        float combined = -3.402823466e+38F;
        if (output < LEO_OUTPUTS) {
            float neural = output_bias[output];
            const unsigned int count = *active_count;
            for (unsigned int index = 0U; index < count; ++index) {
                const unsigned int neuron = active[index];
                neural += output_weights[(unsigned long long)neuron * LEO_OUTPUTS + output] * active_value[index];
            }
            combined = neural;
            const unsigned long long context_row = (unsigned long long)output * embedding_dim;
            for (unsigned int dimension = 0U; dimension < embedding_dim; ++dimension) {
                combined += context_output_weights[context_row + dimension] * latent[dimension];
            }
            neural_logits[output] = neural;
            logits[output] = combined;
        }
        reduction[output] = combined;
    }
    __syncthreads();
    if (lane < 256U) reduction[lane] = fmaxf(reduction[lane], reduction[lane + 256U]);
    __syncthreads();
    for (unsigned int stride = 128U; stride > 0U; stride >>= 1U) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];
    for (unsigned int output = lane; output < 512U; output += blockDim.x) {
        reduction[output] = output < LEO_OUTPUTS ? expf(logits[output] - maximum) : 0.0f;
    }
    __syncthreads();
    if (lane < 256U) reduction[lane] += reduction[lane + 256U];
    __syncthreads();
    for (unsigned int stride = 128U; stride > 0U; stride >>= 1U) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    const float sum = reduction[0];
    for (unsigned int output = lane; output < LEO_OUTPUTS; output += blockDim.x) {
        const bool valid = sum == sum && sum > 0.0f && sum < 3.402823466e+38F;
        const float probability = valid ? expf(logits[output] - maximum) / sum : 1.0f / (float)LEO_OUTPUTS;
        probabilities[output] = probability;
        errors[output] = target_index >= 0 ? probability - (output == (unsigned int)target_index ? 1.0f : 0.0f) : 0.0f;
    }
    __syncthreads();
    if (lane < embedding_dim) {
        float gradient = 0.0f;
        if (target_index >= 0) {
            for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) {
                gradient += context_output_weights[(unsigned long long)output * embedding_dim + lane] * errors[output];
            }
        }
        context_gradient[lane] = gradient;
    }
}

__device__ void leo_p_learning_signals_work(const unsigned long long* p, unsigned long long tick,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const float* output_weights = leo_p_cptr<float>(p, LEO_P_OUTPUT_WEIGHT);
    const float* errors = leo_p_cptr<float>(p, LEO_P_ERRORS);
    const unsigned long long* destination_epoch = leo_p_cptr<unsigned long long>(p, LEO_P_LEARNING_DESTINATION_EPOCH);
    const unsigned int* destination_list = leo_p_cptr<unsigned int>(p, LEO_P_LEARNING_DESTINATION_LIST);
    const unsigned int* destination_count = leo_p_cptr<unsigned int>(p, LEO_P_LEARNING_DESTINATION_COUNT);
    float* learning_signal = leo_p_ptr<float>(p, LEO_P_LEARNING_SIGNAL);
    const unsigned long long tag = tick + 1ULL;
    const unsigned int count = *destination_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int neuron = destination_list[position];
        if (neuron >= cfg->neuron_count || destination_epoch[neuron] != tag) continue;
        const unsigned long long row = (unsigned long long)neuron * LEO_OUTPUTS;
        float signal = 0.0f;
        for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) {
            signal += output_weights[row + output] * errors[output];
        }
        learning_signal[neuron] = signal;
    }

}

__device__ void leo_p_learning_signals_worklist(
    const unsigned long long* p,
    unsigned long long tick
) {
    leo_p_learning_signals_work(
        p, tick, leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_learning_signals(const unsigned long long* p, unsigned long long tick) {
    leo_p_learning_signals_worklist(p, tick);
}

__device__ void leo_p_update_output_work(const unsigned long long* p, float supervised_strength,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    float* output_weights = leo_p_ptr<float>(p, LEO_P_OUTPUT_WEIGHT);
    float* output_bias = leo_p_ptr<float>(p, LEO_P_OUTPUT_BIAS);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const float* active_value = leo_p_cptr<float>(p, LEO_P_ACTIVE_VALUE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* errors = leo_p_cptr<float>(p, LEO_P_ERRORS);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_OUTPUT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_OUTPUT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_OUTPUT_COUNT);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int count = *active_count;
    const unsigned int weight_count = count * LEO_OUTPUTS;
    const unsigned int total = weight_count + LEO_OUTPUTS;
    for (unsigned int index = thread; index < total; index += stride) {
        if (index < weight_count) {
            const unsigned int active_index = index / LEO_OUTPUTS;
            const unsigned int output = index - active_index * LEO_OUTPUTS;
            const unsigned int neuron = active[active_index];
            const unsigned long long slot = (unsigned long long)neuron * LEO_OUTPUTS + output;
            const float delta = leo_clip(
                -cfg->output_learning_rate * supervised_strength * errors[output] * active_value[active_index],
                cfg->max_update
            );
            const float updated = leo_clamp(output_weights[slot] + delta, cfg->weight_min, cfg->weight_max);
            if (!isfinite(updated)) atomicCAS(error_flag, 0U, 1U);
            else output_weights[slot] = updated;
            if (output == 0U) leo_mark_changed(neuron, changed_marks, changed_list, changed_count);
        } else {
            const unsigned int bias_index = index - weight_count;
            const float delta = leo_clip(
                -cfg->output_learning_rate * supervised_strength * errors[bias_index], cfg->max_update
            );
            const float updated = output_bias[bias_index] + delta;
            if (!isfinite(updated)) atomicCAS(error_flag, 0U, 2U);
            else output_bias[bias_index] = updated;
        }
    }

}

__device__ void leo_p_update_output(
    const unsigned long long* p,
    float supervised_strength
) {
    leo_p_update_output_work(
        p, supervised_strength, leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_update_context_work(const unsigned long long* p, float supervised_strength,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    float* context_output_weights = leo_p_ptr<float>(p, LEO_P_CONTEXT_OUTPUT_WEIGHT);
    float* context_embeddings = leo_p_ptr<float>(p, LEO_P_CONTEXT_EMBEDDINGS);
    unsigned int* observations = leo_p_ptr<unsigned int>(p, LEO_P_CONTEXT_OBSERVATIONS);
    const unsigned int* slots = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_SLOTS);
    const float* scales = leo_p_cptr<float>(p, LEO_P_ACTIVE_CONTEXT_SCALES);
    const unsigned int* context_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_COUNT);
    const float* latent = leo_p_cptr<float>(p, LEO_P_CONTEXT_LATENT);
    const float* gradient = leo_p_cptr<float>(p, LEO_P_CONTEXT_GRADIENT);
    const float* errors = leo_p_cptr<float>(p, LEO_P_ERRORS);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_CONTEXT_COUNT);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int embedding_dim = cfg->context_embedding_dim;
    const unsigned int projection_count = LEO_OUTPUTS * embedding_dim;
    const unsigned int embedding_count = (*context_count) * embedding_dim;
    const unsigned int total = projection_count + embedding_count;
    for (unsigned int index = thread; index < total; index += stride) {
        if (index < projection_count) {
            const unsigned int output = index / embedding_dim;
            const unsigned int dimension = index - output * embedding_dim;
            const float g = errors[output] * latent[dimension];
            const float delta = leo_clip(-cfg->context_learning_rate * supervised_strength * g, cfg->max_update);
            const float updated = leo_clamp(context_output_weights[index] + delta, cfg->weight_min, cfg->weight_max);
            if (!isfinite(updated)) atomicCAS(error_flag, 0U, 3U);
            else context_output_weights[index] = updated;
        } else {
            const unsigned int embedding_index = index - projection_count;
            const unsigned int context_index = embedding_index / embedding_dim;
            const unsigned int dimension = embedding_index - context_index * embedding_dim;
            const unsigned int slot = slots[context_index];
            const unsigned long long target = (unsigned long long)slot * embedding_dim + dimension;
            const float g = scales[context_index] * gradient[dimension];
            const float delta = leo_clip(-cfg->context_learning_rate * supervised_strength * g, cfg->max_update);
            const float updated = leo_clamp(context_embeddings[target] + delta, cfg->weight_min, cfg->weight_max);
            if (!isfinite(updated)) atomicCAS(error_flag, 0U, 4U);
            else context_embeddings[target] = updated;
            if (dimension == 0U) {
                leo_mark_changed(slot, changed_marks, changed_list, changed_count);
                leo_atomic_saturating_increment(&observations[slot]);
            }
        }
    }

}

__device__ void leo_p_update_context(
    const unsigned long long* p,
    float supervised_strength
) {
    leo_p_update_context_work(
        p, supervised_strength, leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_update_recurrent_weights_work(
    const unsigned long long* p,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    float supervised_strength,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* target = leo_p_cptr<unsigned int>(p, LEO_P_RECURRENT_TARGET);
    const unsigned char* neuron_type = leo_p_cptr<unsigned char>(p, LEO_P_NEURON_TYPE);
    float* weight = leo_p_ptr<float>(p, LEO_P_RECURRENT_WEIGHT);
    const float* eligibility = leo_p_cptr<float>(p, LEO_P_REC_ELIGIBILITY);
    const float* learning_signal = leo_p_cptr<float>(p, LEO_P_LEARNING_SIGNAL);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_RECURRENT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_RECURRENT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_RECURRENT_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        const unsigned int destination = target[slot];
        const float e = eligibility[slot];
        const float g = learning_signal[destination] * e;
        const float delta = leo_clip(-cfg->recurrent_learning_rate * supervised_strength * g, cfg->max_update);
        const float old_weight = weight[slot];
        float updated = leo_clamp(old_weight + delta, cfg->weight_min, cfg->weight_max);
        const unsigned int source = slot / cfg->synapses_per_neuron;
        updated = neuron_type[source] == 0U ? fmaxf(updated, 0.0f) : fminf(updated, 0.0f);
        if (!isfinite(updated)) {
            atomicCAS(error_flag, 0U, 5U);
            continue;
        }
        if (updated != old_weight) {
            weight[slot] = updated;
            leo_mark_changed(slot, changed_marks, changed_list, changed_count);
        }
        atomicAdd(&counters->learning_eligibility_abs_sum, fabsf(e));
        atomicAdd(&counters->learning_eligibility_count, 1U);
    }

}

__device__ void leo_p_update_recurrent_weights(
    const unsigned long long* p,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    float supervised_strength
) {
    leo_p_update_recurrent_weights_work(
        p, eligible_list, eligible_count, supervised_strength,
        leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_update_input_weights_work(
    const unsigned long long* p,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    float supervised_strength,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* target = leo_p_cptr<unsigned int>(p, LEO_P_INPUT_TARGET);
    float* weight = leo_p_ptr<float>(p, LEO_P_INPUT_WEIGHT);
    const float* eligibility = leo_p_cptr<float>(p, LEO_P_INPUT_ELIGIBILITY);
    const float* learning_signal = leo_p_cptr<float>(p, LEO_P_LEARNING_SIGNAL);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_INPUT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_INPUT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_INPUT_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int count = *eligible_count;
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        const unsigned int destination = target[slot];
        const float e = eligibility[slot];
        const float g = learning_signal[destination] * e;
        const float delta = leo_clip(-cfg->recurrent_learning_rate * supervised_strength * g, cfg->max_update);
        const float old_weight = weight[slot];
        const float updated = leo_clamp(old_weight + delta, 0.0f, cfg->weight_max);
        if (!isfinite(updated)) {
            atomicCAS(error_flag, 0U, 6U);
            continue;
        }
        if (updated != old_weight) {
            weight[slot] = updated;
            leo_mark_changed(slot, changed_marks, changed_list, changed_count);
        }
        atomicAdd(&counters->learning_eligibility_abs_sum, fabsf(e));
        atomicAdd(&counters->learning_eligibility_count, 1U);
    }

}

__device__ void leo_p_update_input_weights(
    const unsigned long long* p,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    float supervised_strength
) {
    leo_p_update_input_weights_work(
        p, eligible_list, eligible_count, supervised_strength,
        leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_inhibitory_homeostasis_work(const unsigned long long* p, float strength,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const float* active_value = leo_p_cptr<float>(p, LEO_P_ACTIVE_VALUE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* activation = leo_p_cptr<float>(p, LEO_P_ACTIVATION);
    const unsigned char* neuron_type = leo_p_cptr<unsigned char>(p, LEO_P_NEURON_TYPE);
    const unsigned int* recurrent_target = leo_p_cptr<unsigned int>(p, LEO_P_RECURRENT_TARGET);
    float* recurrent_weight = leo_p_ptr<float>(p, LEO_P_RECURRENT_WEIGHT);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_RECURRENT_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_RECURRENT_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_RECURRENT_COUNT);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int work = (*active_count) * cfg->synapses_per_neuron;
    for (unsigned int index = thread; index < work; index += stride) {
        const unsigned int active_index = index / cfg->synapses_per_neuron;
        const unsigned int local = index - active_index * cfg->synapses_per_neuron;
        const unsigned int source = active[active_index];
        if (neuron_type[source] != 0U) {
            const unsigned int slot = source * cfg->synapses_per_neuron + local;
            const unsigned int target = recurrent_target[slot];
            const float target_activity = activation[target] > 0.0f ? 1.0f : 0.0f;
            const float activity_error = target_activity - cfg->target_activity;
            const float delta = leo_clip(
                -cfg->inhibitory_learning_rate * strength * active_value[active_index] * activity_error,
                cfg->max_update
            );
            const float old_weight = recurrent_weight[slot];
            const float updated = leo_clamp(old_weight + delta, cfg->weight_min, 0.0f);
            if (!isfinite(updated)) atomicCAS(error_flag, 0U, 7U);
            else if (updated != old_weight) {
                recurrent_weight[slot] = updated;
                leo_mark_changed(slot, changed_marks, changed_list, changed_count);
            }
        }
    }

}

__device__ void leo_p_inhibitory_homeostasis(
    const unsigned long long* p,
    float strength
) {
    leo_p_inhibitory_homeostasis_work(
        p, strength, leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ void leo_p_homeostasis_work(const unsigned long long* p, unsigned long long tick,
    unsigned int thread,
    unsigned int stride
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* activation = leo_p_cptr<float>(p, LEO_P_ACTIVATION);
    const unsigned long long* touched = leo_p_cptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    float* threshold = leo_p_ptr<float>(p, LEO_P_THRESHOLD);
    unsigned int* changed_marks = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_THRESHOLD_MARKS);
    unsigned int* changed_list = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_THRESHOLD_LIST);
    unsigned int* changed_count = leo_p_ptr<unsigned int>(p, LEO_P_CHANGED_THRESHOLD_COUNT);
    const float population_activity = (float)(*active_count) / (float)(cfg->neuron_count == 0U ? 1U : cfg->neuron_count);
    const float population_error = population_activity - cfg->target_activity;
    for (unsigned int neuron = thread; neuron < cfg->neuron_count; neuron += stride) {
        if (touched[neuron] != tick + 1ULL) continue;
        const float local_activity = activation[neuron] > 0.0f ? 1.0f : 0.0f;
        const float local_error = local_activity - cfg->target_activity;
        const float old_threshold = threshold[neuron];
        const float updated = leo_clamp(
            old_threshold + cfg->threshold_homeostasis_rate * (local_error + population_error),
            0.05f, 2.0f
        );
        if (updated != old_threshold) {
            threshold[neuron] = updated;
            leo_mark_changed(neuron, changed_marks, changed_list, changed_count);
        }
    }

}

__device__ void leo_p_homeostasis(
    const unsigned long long* p,
    unsigned long long tick
) {
    leo_p_homeostasis_work(
        p, tick, leo_p_global_thread(), leo_p_global_stride()
    );
}


__device__ __forceinline__ void leo_d_mark(
    unsigned int index,
    unsigned int* marks,
    unsigned int* list,
    unsigned int* count
) {
    leo_mark_changed(index, marks, list, count);
}

__device__ void leo_p_accumulate_output_delta(
    const unsigned long long* p,
    const unsigned long long* d,
    float supervised_strength,
    float batch_scale
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const float* output_weights = leo_p_cptr<float>(p, LEO_P_OUTPUT_WEIGHT);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const float* active_value = leo_p_cptr<float>(p, LEO_P_ACTIVE_VALUE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* errors = leo_p_cptr<float>(p, LEO_P_ERRORS);
    float* delta_weights = leo_p_ptr<float>(d, LEO_D_OUTPUT);
    unsigned int* delta_marks = leo_p_ptr<unsigned int>(d, LEO_D_OUTPUT_MARKS);
    unsigned int* delta_list = leo_p_ptr<unsigned int>(d, LEO_D_OUTPUT_LIST);
    unsigned int* delta_count = leo_p_ptr<unsigned int>(d, LEO_D_OUTPUT_COUNT);
    float* delta_bias = leo_p_ptr<float>(d, LEO_D_OUTPUT_BIAS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int count = *active_count;
    const unsigned int weight_count = count * LEO_OUTPUTS;
    const unsigned int total = weight_count + LEO_OUTPUTS;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();

    for (unsigned int index = thread; index < total; index += stride) {
        if (index < weight_count) {
            const unsigned int active_index = index / LEO_OUTPUTS;
            const unsigned int output = index - active_index * LEO_OUTPUTS;
            const unsigned int neuron = active[active_index];
            const unsigned long long slot = (unsigned long long)neuron * LEO_OUTPUTS + output;
            const float raw_delta = leo_clip(
                -cfg->output_learning_rate * supervised_strength * errors[output] * active_value[active_index],
                cfg->max_update
            );
            const float old_weight = output_weights[slot];
            const float candidate = leo_clamp(old_weight + raw_delta, cfg->weight_min, cfg->weight_max);
            if (!isfinite(candidate)) {
                atomicCAS(error_flag, 0U, 1U);
                continue;
            }
            atomicAdd(&delta_weights[slot], (candidate - old_weight) * batch_scale);
            if (output == 0U) leo_d_mark(neuron, delta_marks, delta_list, delta_count);
        } else {
            const unsigned int output = index - weight_count;
            const float raw_delta = leo_clip(
                -cfg->output_learning_rate * supervised_strength * errors[output],
                cfg->max_update
            );
            if (!isfinite(raw_delta)) atomicCAS(error_flag, 0U, 2U);
            else atomicAdd(&delta_bias[output], raw_delta * batch_scale);
        }
    }
}

__device__ void leo_p_accumulate_context_delta(
    const unsigned long long* p,
    const unsigned long long* d,
    float supervised_strength,
    float batch_scale
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const float* context_output_weights = leo_p_cptr<float>(p, LEO_P_CONTEXT_OUTPUT_WEIGHT);
    const float* context_embeddings = leo_p_cptr<float>(p, LEO_P_CONTEXT_EMBEDDINGS);
    const unsigned int* slots = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_SLOTS);
    const float* scales = leo_p_cptr<float>(p, LEO_P_ACTIVE_CONTEXT_SCALES);
    const unsigned int* context_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_CONTEXT_COUNT);
    const float* latent = leo_p_cptr<float>(p, LEO_P_CONTEXT_LATENT);
    const float* gradient = leo_p_cptr<float>(p, LEO_P_CONTEXT_GRADIENT);
    const float* errors = leo_p_cptr<float>(p, LEO_P_ERRORS);
    float* delta_output = leo_p_ptr<float>(d, LEO_D_CONTEXT_OUTPUT);
    float* delta_embedding = leo_p_ptr<float>(d, LEO_D_CONTEXT_EMBEDDING);
    unsigned int* delta_marks = leo_p_ptr<unsigned int>(d, LEO_D_CONTEXT_MARKS);
    unsigned int* delta_list = leo_p_ptr<unsigned int>(d, LEO_D_CONTEXT_LIST);
    unsigned int* delta_count = leo_p_ptr<unsigned int>(d, LEO_D_CONTEXT_COUNT);
    unsigned int* delta_observations = leo_p_ptr<unsigned int>(d, LEO_D_CONTEXT_OBSERVATIONS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int embedding_dim = cfg->context_embedding_dim;
    const unsigned int projection_count = LEO_OUTPUTS * embedding_dim;
    const unsigned int embedding_count = (*context_count) * embedding_dim;
    const unsigned int total = projection_count + embedding_count;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();

    for (unsigned int index = thread; index < total; index += stride) {
        if (index < projection_count) {
            const unsigned int output = index / embedding_dim;
            const unsigned int dimension = index - output * embedding_dim;
            const float g = errors[output] * latent[dimension];
            const float raw_delta = leo_clip(
                -cfg->context_learning_rate * supervised_strength * g,
                cfg->max_update
            );
            const float old_weight = context_output_weights[index];
            const float candidate = leo_clamp(old_weight + raw_delta, cfg->weight_min, cfg->weight_max);
            if (!isfinite(candidate)) atomicCAS(error_flag, 0U, 3U);
            else atomicAdd(&delta_output[index], (candidate - old_weight) * batch_scale);
        } else {
            const unsigned int embedding_index = index - projection_count;
            const unsigned int context_index = embedding_index / embedding_dim;
            const unsigned int dimension = embedding_index - context_index * embedding_dim;
            const unsigned int slot = slots[context_index];
            const unsigned long long target = (unsigned long long)slot * embedding_dim + dimension;
            const float g = scales[context_index] * gradient[dimension];
            const float raw_delta = leo_clip(
                -cfg->context_learning_rate * supervised_strength * g,
                cfg->max_update
            );
            const float old_value = context_embeddings[target];
            const float candidate = leo_clamp(old_value + raw_delta, cfg->weight_min, cfg->weight_max);
            if (!isfinite(candidate)) atomicCAS(error_flag, 0U, 4U);
            else atomicAdd(&delta_embedding[target], (candidate - old_value) * batch_scale);
            if (dimension == 0U) {
                leo_d_mark(slot, delta_marks, delta_list, delta_count);
                atomicAdd(&delta_observations[slot], 1U);
            }
        }
    }
}

__device__ void leo_p_accumulate_recurrent_delta(
    const unsigned long long* p,
    const unsigned long long* d,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    float supervised_strength,
    float batch_scale
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* target = leo_p_cptr<unsigned int>(p, LEO_P_RECURRENT_TARGET);
    const unsigned char* neuron_type = leo_p_cptr<unsigned char>(p, LEO_P_NEURON_TYPE);
    const float* weight = leo_p_cptr<float>(p, LEO_P_RECURRENT_WEIGHT);
    const float* eligibility = leo_p_cptr<float>(p, LEO_P_REC_ELIGIBILITY);
    const float* learning_signal = leo_p_cptr<float>(p, LEO_P_LEARNING_SIGNAL);
    float* delta_weight = leo_p_ptr<float>(d, LEO_D_RECURRENT);
    unsigned int* delta_marks = leo_p_ptr<unsigned int>(d, LEO_D_RECURRENT_MARKS);
    unsigned int* delta_list = leo_p_ptr<unsigned int>(d, LEO_D_RECURRENT_LIST);
    unsigned int* delta_count = leo_p_ptr<unsigned int>(d, LEO_D_RECURRENT_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int count = *eligible_count;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();

    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        const unsigned int destination = target[slot];
        const float e = eligibility[slot];
        const float g = learning_signal[destination] * e;
        const float raw_delta = leo_clip(
            -cfg->recurrent_learning_rate * supervised_strength * g,
            cfg->max_update
        );
        const float old_weight = weight[slot];
        float candidate = leo_clamp(old_weight + raw_delta, cfg->weight_min, cfg->weight_max);
        const unsigned int source = slot / cfg->synapses_per_neuron;
        candidate = neuron_type[source] == 0U ? fmaxf(candidate, 0.0f) : fminf(candidate, 0.0f);
        if (!isfinite(candidate)) atomicCAS(error_flag, 0U, 5U);
        else if (candidate != old_weight) {
            atomicAdd(&delta_weight[slot], (candidate - old_weight) * batch_scale);
            leo_d_mark(slot, delta_marks, delta_list, delta_count);
        }
        atomicAdd(&counters->learning_eligibility_abs_sum, fabsf(e));
        atomicAdd(&counters->learning_eligibility_count, 1U);
    }
}

__device__ void leo_p_accumulate_input_delta(
    const unsigned long long* p,
    const unsigned long long* d,
    const unsigned int* eligible_list,
    const unsigned int* eligible_count,
    float supervised_strength,
    float batch_scale
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* target = leo_p_cptr<unsigned int>(p, LEO_P_INPUT_TARGET);
    const float* weight = leo_p_cptr<float>(p, LEO_P_INPUT_WEIGHT);
    const float* eligibility = leo_p_cptr<float>(p, LEO_P_INPUT_ELIGIBILITY);
    const float* learning_signal = leo_p_cptr<float>(p, LEO_P_LEARNING_SIGNAL);
    float* delta_weight = leo_p_ptr<float>(d, LEO_D_INPUT);
    unsigned int* delta_marks = leo_p_ptr<unsigned int>(d, LEO_D_INPUT_MARKS);
    unsigned int* delta_list = leo_p_ptr<unsigned int>(d, LEO_D_INPUT_LIST);
    unsigned int* delta_count = leo_p_ptr<unsigned int>(d, LEO_D_INPUT_COUNT);
    LeoStepCounters* counters = leo_p_ptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int count = *eligible_count;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();

    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int slot = eligible_list[position];
        const unsigned int destination = target[slot];
        const float e = eligibility[slot];
        const float g = learning_signal[destination] * e;
        const float raw_delta = leo_clip(
            -cfg->recurrent_learning_rate * supervised_strength * g,
            cfg->max_update
        );
        const float old_weight = weight[slot];
        const float candidate = leo_clamp(old_weight + raw_delta, 0.0f, cfg->weight_max);
        if (!isfinite(candidate)) atomicCAS(error_flag, 0U, 6U);
        else if (candidate != old_weight) {
            atomicAdd(&delta_weight[slot], (candidate - old_weight) * batch_scale);
            leo_d_mark(slot, delta_marks, delta_list, delta_count);
        }
        atomicAdd(&counters->learning_eligibility_abs_sum, fabsf(e));
        atomicAdd(&counters->learning_eligibility_count, 1U);
    }
}

__device__ void leo_p_accumulate_inhibitory_delta(
    const unsigned long long* p,
    const unsigned long long* d,
    float strength,
    float batch_scale
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* active = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE);
    const float* active_value = leo_p_cptr<float>(p, LEO_P_ACTIVE_VALUE);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* activation = leo_p_cptr<float>(p, LEO_P_ACTIVATION);
    const unsigned char* neuron_type = leo_p_cptr<unsigned char>(p, LEO_P_NEURON_TYPE);
    const unsigned int* recurrent_target = leo_p_cptr<unsigned int>(p, LEO_P_RECURRENT_TARGET);
    const float* recurrent_weight = leo_p_cptr<float>(p, LEO_P_RECURRENT_WEIGHT);
    float* delta_weight = leo_p_ptr<float>(d, LEO_D_RECURRENT);
    unsigned int* delta_marks = leo_p_ptr<unsigned int>(d, LEO_D_RECURRENT_MARKS);
    unsigned int* delta_list = leo_p_ptr<unsigned int>(d, LEO_D_RECURRENT_LIST);
    unsigned int* delta_count = leo_p_ptr<unsigned int>(d, LEO_D_RECURRENT_COUNT);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    const unsigned int work = (*active_count) * cfg->synapses_per_neuron;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();

    for (unsigned int index = thread; index < work; index += stride) {
        const unsigned int active_index = index / cfg->synapses_per_neuron;
        const unsigned int local = index - active_index * cfg->synapses_per_neuron;
        const unsigned int source = active[active_index];
        if (neuron_type[source] == 0U) continue;
        const unsigned int slot = source * cfg->synapses_per_neuron + local;
        const unsigned int destination = recurrent_target[slot];
        const float target_activity = activation[destination] > 0.0f ? 1.0f : 0.0f;
        const float activity_error = target_activity - cfg->target_activity;
        const float raw_delta = leo_clip(
            -cfg->inhibitory_learning_rate * strength * active_value[active_index] * activity_error,
            cfg->max_update
        );
        const float old_weight = recurrent_weight[slot];
        const float candidate = leo_clamp(old_weight + raw_delta, cfg->weight_min, 0.0f);
        if (!isfinite(candidate)) atomicCAS(error_flag, 0U, 7U);
        else if (candidate != old_weight) {
            atomicAdd(&delta_weight[slot], (candidate - old_weight) * batch_scale);
            leo_d_mark(slot, delta_marks, delta_list, delta_count);
        }
    }
}

__device__ void leo_p_accumulate_homeostasis_delta_worklist(
    const unsigned long long* p,
    const unsigned long long* d,
    unsigned long long tick,
    float batch_scale
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* activation = leo_p_cptr<float>(p, LEO_P_ACTIVATION);
    const unsigned long long* touched = leo_p_cptr<unsigned long long>(p, LEO_P_TOUCHED_EPOCH);
    const unsigned int* touched_list = leo_p_cptr<unsigned int>(p, LEO_P_TOUCHED_LIST);
    const unsigned int* touched_count = leo_p_cptr<unsigned int>(p, LEO_P_TOUCHED_COUNT);
    const float* threshold = leo_p_cptr<float>(p, LEO_P_THRESHOLD);
    float* delta_threshold = leo_p_ptr<float>(d, LEO_D_THRESHOLD);
    unsigned int* delta_marks = leo_p_ptr<unsigned int>(d, LEO_D_THRESHOLD_MARKS);
    unsigned int* delta_list = leo_p_ptr<unsigned int>(d, LEO_D_THRESHOLD_LIST);
    unsigned int* delta_count = leo_p_ptr<unsigned int>(d, LEO_D_THRESHOLD_COUNT);
    const float population_activity = (float)(*active_count) / (float)(cfg->neuron_count == 0U ? 1U : cfg->neuron_count);
    const float population_error = population_activity - cfg->target_activity;
    const unsigned long long tag = tick + 1ULL;
    const unsigned int count = *touched_count;
    const unsigned int thread = leo_p_global_thread();
    const unsigned int stride = leo_p_global_stride();
    for (unsigned int position = thread; position < count; position += stride) {
        const unsigned int neuron = touched_list[position];
        if (neuron >= cfg->neuron_count || touched[neuron] != tag) continue;
        const float local_activity = activation[neuron] > 0.0f ? 1.0f : 0.0f;
        const float local_error = local_activity - cfg->target_activity;
        const float old_threshold = threshold[neuron];
        const float candidate = leo_clamp(
            old_threshold + cfg->threshold_homeostasis_rate * (local_error + population_error),
            0.05f,
            2.0f
        );
        if (candidate != old_threshold) {
            atomicAdd(&delta_threshold[neuron], (candidate - old_threshold) * batch_scale);
            leo_d_mark(neuron, delta_marks, delta_list, delta_count);
        }
    }
}

__device__ void leo_p_accumulate_homeostasis_delta(
    const unsigned long long* p,
    const unsigned long long* d,
    unsigned long long tick,
    float batch_scale
) {
    leo_p_accumulate_homeostasis_delta_worklist(p, d, tick, batch_scale);
}

__device__ void leo_p_capture_training_step(
    const unsigned long long* p,
    int target_index,
    unsigned int record_index
) {
    if (leo_p_global_thread() != 0U) return;
    const float* probabilities = leo_p_cptr<float>(p, LEO_P_PROBABILITIES);
    const float* neural_logits = leo_p_cptr<float>(p, LEO_P_NEURAL_LOGITS);
    const unsigned int* active_count = leo_p_cptr<unsigned int>(p, LEO_P_ACTIVE_COUNT);
    const float* population_inhibition = leo_p_cptr<float>(p, LEO_P_POPULATION_INHIBITION);
    const LeoStepCounters* counters = leo_p_cptr<LeoStepCounters>(p, LEO_P_COUNTERS);
    unsigned int* error_flag = leo_p_ptr<unsigned int>(p, LEO_P_ERROR_FLAG);
    LeoTrainingStepRecord* records = leo_p_ptr<LeoTrainingStepRecord>(p, LEO_P_TRAINING_STEP_RECORDS);
    LeoTrainingStepRecord record;
    unsigned int predicted = 0U;
    float best = probabilities[0];
    for (unsigned int output = 1U; output < LEO_OUTPUTS; ++output) {
        const float candidate = probabilities[output];
        if (candidate > best) {
            best = candidate;
            predicted = output;
        }
    }
    float loss = 0.0f;
    float neural_loss = 0.0f;
    if (target_index >= 0) {
        const float probability = fmaxf(probabilities[(unsigned int)target_index], 1.0e-12f);
        loss = -logf(probability);
        float maximum = neural_logits[0];
        for (unsigned int output = 1U; output < LEO_OUTPUTS; ++output) maximum = fmaxf(maximum, neural_logits[output]);
        float sum = 0.0f;
        for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) sum += expf(neural_logits[output] - maximum);
        neural_loss = (sum == sum && sum > 0.0f && sum < 3.402823466e+38F)
            ? maximum + logf(sum) - neural_logits[(unsigned int)target_index]
            : logf((float)LEO_OUTPUTS);
    }
    record.loss = loss;
    record.neural_loss = neural_loss;
    record.population_inhibition = *population_inhibition;
    record.learning_eligibility_abs_sum = counters->learning_eligibility_abs_sum;
    record.predicted_index = predicted;
    record.active_count = *active_count;
    record.suprathreshold = counters->suprathreshold;
    record.block_selected = counters->block_selected;
    record.global_clipped = counters->global_clipped;
    record.context_cells = counters->context_cells;
    record.context_probes = counters->context_probes;
    record.eligible_recurrent = counters->eligible_recurrent;
    record.eligible_input = counters->eligible_input;
    record.learning_eligibility_count = counters->learning_eligibility_count;
    record.error_code = *error_flag;
    *error_flag = 0U;
    records[record_index] = record;
}

__device__ void leo_train_story_block(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned int learning_trace_raw,
    float strength,
    float* shared_values,
    unsigned int* shared_neurons,
    float* shared_latent,
    float* shared_reduction
) {
    const bool learning_trace = learning_trace_raw != 0U;

    for (unsigned int step_index = 0U; step_index < step_count; ++step_index) {
        const LeoPersistentStep step = steps[step_index];
        const unsigned long long tick = base_tick + (unsigned long long)step_index;
        const bool context_enabled = step.context_enabled != 0U;
        unsigned int* rec_a_list = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_LIST);
        unsigned int* rec_a_count = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_COUNT);
        unsigned int* rec_b_list = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_NEXT_ELIGIBLE_LIST);
        unsigned int* rec_b_count = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_NEXT_ELIGIBLE_COUNT);
        unsigned int* input_a_list = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_LIST);
        unsigned int* input_a_count = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_COUNT);
        unsigned int* input_b_list = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_NEXT_ELIGIBLE_LIST);
        unsigned int* input_b_count = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_NEXT_ELIGIBLE_COUNT);
        const bool even = (step_index & 1U) == 0U;
        unsigned int* rec_current_list = even ? rec_a_list : rec_b_list;
        unsigned int* rec_current_count = even ? rec_a_count : rec_b_count;
        unsigned int* rec_next_list = even ? rec_b_list : rec_a_list;
        unsigned int* rec_next_count = even ? rec_b_count : rec_a_count;
        unsigned int* input_current_list = even ? input_a_list : input_b_list;
        unsigned int* input_current_count = even ? input_a_count : input_b_count;
        unsigned int* input_next_list = even ? input_b_list : input_a_list;
        unsigned int* input_next_count = even ? input_b_count : input_a_count;

        leo_p_start_tick(pointers, tick);
        __syncthreads();
        leo_p_deliver_events(pointers, tick, learning_trace, rec_current_list, rec_current_count);
        __syncthreads();
        leo_p_inject_symbol(
            pointers, tick, step.symbol, learning_trace, input_current_list, input_current_count
        );
        __syncthreads();
        leo_p_context_resolve(pointers, step.symbol, context_enabled, learning_trace && context_enabled);
        __syncthreads();
        leo_p_select_blocks(pointers, tick);
        __syncthreads();
        leo_p_select_global(pointers, tick, shared_values, shared_neurons);
        __syncthreads();
        leo_p_cache_surrogate(pointers, tick);
        __syncthreads();

        if (learning_trace) {
            if (threadIdx.x == 0U) {
                *rec_next_count = 0U;
                *input_next_count = 0U;
            }
            __syncthreads();
            leo_p_update_recurrent_eligibility(
                pointers, tick, rec_current_list, rec_current_count, rec_next_list, rec_next_count
            );
            __syncthreads();
            leo_p_update_input_eligibility(
                pointers, tick, input_current_list, input_current_count, input_next_list, input_next_count
            );
            __syncthreads();
        }

        leo_p_post_and_emit(pointers, tick);
        __syncthreads();
        leo_p_forward(pointers, step.target_index, shared_latent, shared_reduction);
        __syncthreads();

        if (step.target_index >= 0 && strength > 0.0f) {
            leo_p_learning_signals(pointers, tick);
            __syncthreads();
            leo_p_update_output(pointers, step.supervised_strength);
            __syncthreads();
            if (context_enabled) {
                leo_p_update_context(pointers, step.supervised_strength);
                __syncthreads();
            }
            const unsigned int* learning_rec_list = learning_trace ? rec_next_list : rec_current_list;
            const unsigned int* learning_rec_count = learning_trace ? rec_next_count : rec_current_count;
            const unsigned int* learning_input_list = learning_trace ? input_next_list : input_current_list;
            const unsigned int* learning_input_count = learning_trace ? input_next_count : input_current_count;
            leo_p_update_recurrent_weights(
                pointers, learning_rec_list, learning_rec_count, step.supervised_strength
            );
            __syncthreads();
            leo_p_update_input_weights(
                pointers, learning_input_list, learning_input_count, step.supervised_strength
            );
            __syncthreads();
            leo_p_inhibitory_homeostasis(pointers, strength);
            __syncthreads();
        }

        if (learning_trace) {
            leo_p_homeostasis(pointers, tick);
            __syncthreads();
        }
        leo_p_capture_training_step(pointers, step.target_index, step_index);
        __syncthreads();
    }
}

__device__ void leo_advance_frozen_story_block(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    float* shared_values,
    unsigned int* shared_neurons
) {
    unsigned int* recurrent_list = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_LIST);
    unsigned int* recurrent_count = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_COUNT);
    unsigned int* input_list = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_LIST);
    unsigned int* input_count = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_COUNT);

    for (unsigned int step_index = 0U; step_index < step_count; ++step_index) {
        const unsigned int symbol = steps[step_index].symbol;
        const unsigned long long tick = base_tick + (unsigned long long)step_index;

        leo_p_start_tick(pointers, tick);
        __syncthreads();
        leo_p_deliver_events(
            pointers, tick, false, recurrent_list, recurrent_count
        );
        __syncthreads();
        leo_p_inject_symbol(
            pointers, tick, symbol, false, input_list, input_count
        );
        __syncthreads();
        leo_p_context_advance_history(pointers, symbol);
        __syncthreads();
        leo_p_select_blocks(pointers, tick);
        __syncthreads();
        leo_p_select_global(pointers, tick, shared_values, shared_neurons);
        __syncthreads();
        leo_p_post_and_emit(pointers, tick);
        __syncthreads();
    }
}

extern "C" __global__ void leo_advance_frozen_persistent(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick
) {
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    if (blockIdx.x != 0U) return;
    leo_advance_frozen_story_block(
        pointers,
        steps,
        step_count,
        base_tick,
        shared_values,
        shared_neurons
    );
}


// Cooperative frozen replay-prefix advancement. Frozen prefix state is exactly
// the same as leo_advance_frozen_persistent, but independent model blocks are
// evaluated by separate CUDA blocks. Grid barriers preserve the original phase
// order, and block-local arithmetic/order within each model block is unchanged.
enum LeoCudaFrozenProfileCounter {
    LEO_CUDA_FROZEN_PROFILE_SAMPLES = 0,
    LEO_CUDA_FROZEN_PROFILE_PRE = 1,
    LEO_CUDA_FROZEN_PROFILE_SELECT_BLOCKS = 2,
    LEO_CUDA_FROZEN_PROFILE_SELECT_GLOBAL = 3,
    LEO_CUDA_FROZEN_PROFILE_POST_EMIT = 4
};

template <bool PROFILE>
__device__ __forceinline__ unsigned long long leo_profile_phase_start() {
    if (PROFILE && blockIdx.x == 0U && threadIdx.x == 0U) return clock64();
    return 0ULL;
}

template <bool PROFILE>
__device__ __forceinline__ void leo_profile_phase_end(
    cooperative_groups::grid_group grid,
    unsigned long long* counters,
    unsigned int counter,
    unsigned long long started
) {
    grid.sync();
    if (PROFILE && blockIdx.x == 0U && threadIdx.x == 0U && counters != nullptr) {
        counters[counter] += clock64() - started;
    }
    // Profiling is diagnostic-only. The extra barrier keeps phase timestamps
    // non-overlapping and is compiled away from the normal kernel.
    if (PROFILE) grid.sync();
}

template <bool PROFILE>
__device__ __forceinline__ void leo_advance_frozen_cooperative_body(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned long long* profile_counters
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(pointers, LEO_P_CONFIG);
    unsigned int* recurrent_list = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_LIST);
    unsigned int* recurrent_count = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_COUNT);
    unsigned int* input_list = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_LIST);
    unsigned int* input_count = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_COUNT);

    for (unsigned int step_index = 0U; step_index < step_count; ++step_index) {
        const unsigned int symbol = steps[step_index].symbol;
        const unsigned long long tick = base_tick + (unsigned long long)step_index;

        unsigned long long phase_started = leo_profile_phase_start<PROFILE>();
        // Event delivery/input injection retain the established one-block
        // atomic/update ordering. Frozen prefix optimization only distributes
        // phases whose rows are independent.
        if (blockIdx.x == 0U) {
            leo_p_start_tick(pointers, tick);
            __syncthreads();
            leo_p_deliver_events(pointers, tick, false, recurrent_list, recurrent_count);
            __syncthreads();
            leo_p_inject_symbol(pointers, tick, symbol, false, input_list, input_count);
            __syncthreads();
            leo_p_context_advance_history(pointers, symbol);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_FROZEN_PROFILE_PRE, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        for (unsigned int model_block = blockIdx.x;
             model_block < cfg->block_count;
             model_block += gridDim.x) {
            leo_p_select_model_block(pointers, tick, model_block);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_FROZEN_PROFILE_SELECT_BLOCKS, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        if (blockIdx.x == 0U) {
            leo_p_select_global(pointers, tick, shared_values, shared_neurons);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_FROZEN_PROFILE_SELECT_GLOBAL, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        leo_p_post_and_emit_grid(pointers, tick);
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_FROZEN_PROFILE_POST_EMIT, phase_started
        );

        if (PROFILE && blockIdx.x == 0U && threadIdx.x == 0U && profile_counters != nullptr) {
            profile_counters[LEO_CUDA_FROZEN_PROFILE_SAMPLES] += 1ULL;
        }
        if (PROFILE) grid.sync();
    }
}

extern "C" __global__ void leo_advance_frozen_cooperative(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick
) {
    leo_advance_frozen_cooperative_body<false>(
        pointers, steps, step_count, base_tick, nullptr
    );
}

extern "C" __global__ void leo_advance_frozen_cooperative_profiled(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned long long* profile_counters
) {
    leo_advance_frozen_cooperative_body<true>(
        pointers, steps, step_count, base_tick, profile_counters
    );
}

enum LeoCudaReplayProfileCounter {
    LEO_CUDA_REPLAY_PROFILE_SAMPLES = 0,
    LEO_CUDA_REPLAY_PROFILE_PRE = 1,
    LEO_CUDA_REPLAY_PROFILE_SELECT_BLOCKS = 2,
    LEO_CUDA_REPLAY_PROFILE_SELECT_GLOBAL = 3,
    LEO_CUDA_REPLAY_PROFILE_CACHE_SURROGATE = 4,
    LEO_CUDA_REPLAY_PROFILE_RECURRENT_ELIGIBILITY = 5,
    LEO_CUDA_REPLAY_PROFILE_INPUT_ELIGIBILITY = 6,
    LEO_CUDA_REPLAY_PROFILE_POST_EMIT = 7,
    LEO_CUDA_REPLAY_PROFILE_FORWARD = 8,
    LEO_CUDA_REPLAY_PROFILE_LEARNING_SIGNALS = 9,
    LEO_CUDA_REPLAY_PROFILE_OUTPUT_UPDATE = 10,
    LEO_CUDA_REPLAY_PROFILE_CONTEXT_UPDATE = 11,
    LEO_CUDA_REPLAY_PROFILE_RECURRENT_UPDATE = 12,
    LEO_CUDA_REPLAY_PROFILE_INPUT_UPDATE = 13,
    LEO_CUDA_REPLAY_PROFILE_INHIBITORY = 14,
    LEO_CUDA_REPLAY_PROFILE_HOMEOSTASIS = 15,
    LEO_CUDA_REPLAY_PROFILE_CAPTURE = 16
};

// Single-story cooperative trainer used by replay and any other supervised
// single-runtime step batch. Order-sensitive event delivery, input injection,
// context resolution, global winner selection, and the forward reduction keep
// their original block-local arithmetic. Independent model blocks, sparse rows,
// destinations, weights, homeostasis rows, and post/emit records use the whole
// cooperative grid. This changes execution geometry only; FP32 equations,
// target order, parameter visibility, and per-step barriers are unchanged.
template <bool PROFILE>
__device__ __forceinline__ void leo_train_cooperative_body(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned int learning_trace_raw,
    float strength,
    unsigned long long* profile_counters
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    __shared__ float shared_latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float shared_reduction[512];
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(pointers, LEO_P_CONFIG);
    const bool learning_trace = learning_trace_raw != 0U;
    const unsigned int grid_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_stride = gridDim.x * blockDim.x;

    for (unsigned int step_index = 0U; step_index < step_count; ++step_index) {
        const LeoPersistentStep step = steps[step_index];
        const unsigned long long tick = base_tick + (unsigned long long)step_index;
        const bool context_enabled = step.context_enabled != 0U;
        const bool supervised = step.target_index >= 0 && strength > 0.0f;
        unsigned int* rec_a_list = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_LIST);
        unsigned int* rec_a_count = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_ELIGIBLE_COUNT);
        unsigned int* rec_b_list = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_NEXT_ELIGIBLE_LIST);
        unsigned int* rec_b_count = leo_p_ptr<unsigned int>(pointers, LEO_P_REC_NEXT_ELIGIBLE_COUNT);
        unsigned int* input_a_list = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_LIST);
        unsigned int* input_a_count = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_ELIGIBLE_COUNT);
        unsigned int* input_b_list = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_NEXT_ELIGIBLE_LIST);
        unsigned int* input_b_count = leo_p_ptr<unsigned int>(pointers, LEO_P_INPUT_NEXT_ELIGIBLE_COUNT);
        const bool even = (step_index & 1U) == 0U;
        unsigned int* rec_current_list = even ? rec_a_list : rec_b_list;
        unsigned int* rec_current_count = even ? rec_a_count : rec_b_count;
        unsigned int* rec_next_list = even ? rec_b_list : rec_a_list;
        unsigned int* rec_next_count = even ? rec_b_count : rec_a_count;
        unsigned int* input_current_list = even ? input_a_list : input_b_list;
        unsigned int* input_current_count = even ? input_a_count : input_b_count;
        unsigned int* input_next_list = even ? input_b_list : input_a_list;
        unsigned int* input_next_count = even ? input_b_count : input_a_count;

        unsigned long long phase_started = leo_profile_phase_start<PROFILE>();
        // Preserve established atomic ordering for event delivery and input
        // injection by keeping this phase on block 0.
        if (blockIdx.x == 0U) {
            leo_p_start_tick(pointers, tick);
            __syncthreads();
            leo_p_deliver_events(
                pointers, tick, learning_trace, rec_current_list, rec_current_count
            );
            __syncthreads();
            leo_p_inject_symbol(
                pointers, tick, step.symbol, learning_trace,
                input_current_list, input_current_count
            );
            __syncthreads();
            leo_p_context_resolve(
                pointers, step.symbol, context_enabled, learning_trace && context_enabled
            );
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_PRE, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        for (unsigned int model_block = blockIdx.x;
             model_block < cfg->block_count;
             model_block += gridDim.x) {
            leo_p_select_model_block(pointers, tick, model_block);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_SELECT_BLOCKS, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        if (blockIdx.x == 0U) {
            leo_p_select_global(pointers, tick, shared_values, shared_neurons);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_SELECT_GLOBAL, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        leo_p_cache_surrogate_work(pointers, tick, grid_thread, grid_stride);
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_CACHE_SURROGATE, phase_started
        );

        if (learning_trace) {
            if (blockIdx.x == 0U && threadIdx.x == 0U) {
                *rec_next_count = 0U;
                *input_next_count = 0U;
            }
            grid.sync();

            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_update_recurrent_eligibility_work(
                pointers, tick, rec_current_list, rec_current_count,
                rec_next_list, rec_next_count, grid_thread, grid_stride
            );
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters,
                LEO_CUDA_REPLAY_PROFILE_RECURRENT_ELIGIBILITY, phase_started
            );

            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_update_input_eligibility_work(
                pointers, tick, input_current_list, input_current_count,
                input_next_list, input_next_count, grid_thread, grid_stride
            );
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters,
                LEO_CUDA_REPLAY_PROFILE_INPUT_ELIGIBILITY, phase_started
            );
        }

        phase_started = leo_profile_phase_start<PROFILE>();
        leo_p_post_and_emit_work(pointers, tick, grid_thread, grid_stride);
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_POST_EMIT, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        if (blockIdx.x == 0U) {
            leo_p_forward(pointers, step.target_index, shared_latent, shared_reduction);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_FORWARD, phase_started
        );

        if (supervised) {
            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_learning_signals_work(pointers, tick, grid_thread, grid_stride);
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters,
                LEO_CUDA_REPLAY_PROFILE_LEARNING_SIGNALS, phase_started
            );

            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_update_output_work(
                pointers, step.supervised_strength, grid_thread, grid_stride
            );
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_OUTPUT_UPDATE, phase_started
            );

            phase_started = leo_profile_phase_start<PROFILE>();
            if (context_enabled) {
                leo_p_update_context_work(
                    pointers, step.supervised_strength, grid_thread, grid_stride
                );
            }
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_CONTEXT_UPDATE, phase_started
            );

            const unsigned int* learning_rec_list =
                learning_trace ? rec_next_list : rec_current_list;
            const unsigned int* learning_rec_count =
                learning_trace ? rec_next_count : rec_current_count;
            const unsigned int* learning_input_list =
                learning_trace ? input_next_list : input_current_list;
            const unsigned int* learning_input_count =
                learning_trace ? input_next_count : input_current_count;

            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_update_recurrent_weights_work(
                pointers, learning_rec_list, learning_rec_count,
                step.supervised_strength, grid_thread, grid_stride
            );
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_RECURRENT_UPDATE, phase_started
            );

            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_update_input_weights_work(
                pointers, learning_input_list, learning_input_count,
                step.supervised_strength, grid_thread, grid_stride
            );
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_INPUT_UPDATE, phase_started
            );

            phase_started = leo_profile_phase_start<PROFILE>();
            leo_p_inhibitory_homeostasis_work(
                pointers, strength, grid_thread, grid_stride
            );
            leo_profile_phase_end<PROFILE>(
                grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_INHIBITORY, phase_started
            );
        }

        phase_started = leo_profile_phase_start<PROFILE>();
        if (learning_trace) {
            leo_p_homeostasis_work(pointers, tick, grid_thread, grid_stride);
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_HOMEOSTASIS, phase_started
        );

        phase_started = leo_profile_phase_start<PROFILE>();
        if (blockIdx.x == 0U) {
            leo_p_capture_training_step(pointers, step.target_index, step_index);
            __syncthreads();
        }
        leo_profile_phase_end<PROFILE>(
            grid, profile_counters, LEO_CUDA_REPLAY_PROFILE_CAPTURE, phase_started
        );

        if (PROFILE && blockIdx.x == 0U && threadIdx.x == 0U && profile_counters != nullptr) {
            profile_counters[LEO_CUDA_REPLAY_PROFILE_SAMPLES] += 1ULL;
        }
        if (PROFILE) grid.sync();
    }
}

extern "C" __global__ void leo_train_cooperative(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned int learning_trace_raw,
    float strength
) {
    leo_train_cooperative_body<false>(
        pointers, steps, step_count, base_tick,
        learning_trace_raw, strength, nullptr
    );
}

extern "C" __global__ void leo_train_cooperative_profiled(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned int learning_trace_raw,
    float strength,
    unsigned long long* profile_counters
) {
    leo_train_cooperative_body<true>(
        pointers, steps, step_count, base_tick,
        learning_trace_raw, strength, profile_counters
    );
}

extern "C" __global__ void leo_train_persistent(
    const unsigned long long* pointers,
    const LeoPersistentStep* steps,
    unsigned int step_count,
    unsigned long long base_tick,
    unsigned int learning_trace_raw,
    float strength
) {
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    __shared__ float shared_latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float shared_reduction[512];
    if (blockIdx.x != 0U) return;
    leo_train_story_block(
        pointers,
        steps,
        step_count,
        base_tick,
        learning_trace_raw,
        strength,
        shared_values,
        shared_neurons,
        shared_latent,
        shared_reduction
    );
}


// Exact shared-wavefront story batch. Static topology/configuration is shared,
// while recurrent state and every mutable learned parameter are lane-private.
// Kernel boundaries on one CUDA stream provide phase ordering; lanes are only
// averaged once after their complete stories finish.
// Shared-model wavefront phase helpers. Both the cooperative fused kernel and
// the capability fallback kernels use these exact bodies.
__device__ __forceinline__ void leo_shared_phase_pre_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw,
    unsigned int lane
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const LeoPersistentStep* steps = reinterpret_cast<const LeoPersistentStep*>(
        step_buffer_addresses[lane]
    );
    const LeoPersistentStep step = steps[step_index];
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    const bool learning_trace = learning_trace_raw != 0U;
    const bool context_enabled = step.context_enabled != 0U;

    unsigned int* rec_a_list = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_LIST);
    unsigned int* rec_a_count = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_COUNT);
    unsigned int* rec_b_list = leo_p_ptr<unsigned int>(p, LEO_P_REC_NEXT_ELIGIBLE_LIST);
    unsigned int* rec_b_count = leo_p_ptr<unsigned int>(p, LEO_P_REC_NEXT_ELIGIBLE_COUNT);
    unsigned int* input_a_list = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_LIST);
    unsigned int* input_a_count = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_COUNT);
    unsigned int* input_b_list = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_NEXT_ELIGIBLE_LIST);
    unsigned int* input_b_count = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_NEXT_ELIGIBLE_COUNT);
    const bool even = (step_index & 1U) == 0U;
    unsigned int* rec_current_list = even ? rec_a_list : rec_b_list;
    unsigned int* rec_current_count = even ? rec_a_count : rec_b_count;
    unsigned int* input_current_list = even ? input_a_list : input_b_list;
    unsigned int* input_current_count = even ? input_a_count : input_b_count;

    leo_p_start_tick(p, tick);
    __syncthreads();
    leo_p_deliver_events(p, tick, learning_trace, rec_current_list, rec_current_count);
    __syncthreads();
    leo_p_inject_symbol(p, tick, step.symbol, learning_trace, input_current_list, input_current_count);
    __syncthreads();
    // Context tables are lane-private in the exact logical-batch path, so use
    // the ordinary single-story resolver including deterministic weakest-slot
    // replacement semantics.
    leo_p_context_resolve(p, step.symbol, context_enabled, learning_trace && context_enabled);
}

__device__ __forceinline__ void leo_shared_phase_select_block(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int model_block_count,
    unsigned int work_block
) {
    if (model_block_count == 0U) return;
    const unsigned int lane = work_block / model_block_count;
    const unsigned int model_block = work_block - lane * model_block_count;
    if (lane >= lane_count || step_index >= step_counts[lane]) return;

    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(p, LEO_P_CONFIG);
    if (model_block >= cfg->block_count) return;
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    leo_p_select_model_block(p, tick, model_block);
}

__device__ __forceinline__ void leo_shared_phase_post_select_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int lane,
    float* shared_values,
    unsigned int* shared_neurons
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    leo_p_select_global(p, tick, shared_values, shared_neurons);
}

__device__ __forceinline__ void leo_shared_phase_cache_surrogate_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int lane
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    leo_p_cache_surrogate_worklist(p, tick);
}

__device__ __forceinline__ void leo_shared_phase_post_core_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw,
    unsigned int lane,
    float* shared_latent,
    float* shared_reduction
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const LeoPersistentStep* steps = reinterpret_cast<const LeoPersistentStep*>(
        step_buffer_addresses[lane]
    );
    const LeoPersistentStep step = steps[step_index];
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    const bool learning_trace = learning_trace_raw != 0U;

    unsigned int* rec_a_list = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_LIST);
    unsigned int* rec_a_count = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_COUNT);
    unsigned int* rec_b_list = leo_p_ptr<unsigned int>(p, LEO_P_REC_NEXT_ELIGIBLE_LIST);
    unsigned int* rec_b_count = leo_p_ptr<unsigned int>(p, LEO_P_REC_NEXT_ELIGIBLE_COUNT);
    unsigned int* input_a_list = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_LIST);
    unsigned int* input_a_count = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_COUNT);
    unsigned int* input_b_list = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_NEXT_ELIGIBLE_LIST);
    unsigned int* input_b_count = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_NEXT_ELIGIBLE_COUNT);
    const bool even = (step_index & 1U) == 0U;
    unsigned int* rec_current_list = even ? rec_a_list : rec_b_list;
    unsigned int* rec_current_count = even ? rec_a_count : rec_b_count;
    unsigned int* rec_next_list = even ? rec_b_list : rec_a_list;
    unsigned int* rec_next_count = even ? rec_b_count : rec_a_count;
    unsigned int* input_current_list = even ? input_a_list : input_b_list;
    unsigned int* input_current_count = even ? input_a_count : input_b_count;
    unsigned int* input_next_list = even ? input_b_list : input_a_list;
    unsigned int* input_next_count = even ? input_b_count : input_a_count;

    if (learning_trace) {
        if (threadIdx.x == 0U) {
            *rec_next_count = 0U;
            *input_next_count = 0U;
        }
        __syncthreads();
        leo_p_update_recurrent_eligibility(
            p, tick, rec_current_list, rec_current_count, rec_next_list, rec_next_count
        );
        __syncthreads();
        leo_p_update_input_eligibility(
            p, tick, input_current_list, input_current_count, input_next_list, input_next_count
        );
        __syncthreads();
    }

    leo_p_post_and_emit(p, tick);
    __syncthreads();
    leo_p_forward(p, step.target_index, shared_latent, shared_reduction);
}

__device__ __forceinline__ void leo_shared_phase_learning_signals_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    float strength,
    unsigned int lane
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const LeoPersistentStep* steps = reinterpret_cast<const LeoPersistentStep*>(
        step_buffer_addresses[lane]
    );
    if (steps[step_index].target_index < 0 || strength <= 0.0f) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    leo_p_learning_signals_worklist(p, tick);
}

__device__ __forceinline__ void leo_shared_phase_post_deltas_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw,
    float strength,
    float batch_scale,
    const unsigned long long* delta_pointers,
    unsigned int lane
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const LeoPersistentStep* steps = reinterpret_cast<const LeoPersistentStep*>(
        step_buffer_addresses[lane]
    );
    const LeoPersistentStep step = steps[step_index];
    if (step.target_index < 0 || strength <= 0.0f) return;
    const bool learning_trace = learning_trace_raw != 0U;
    const bool context_enabled = step.context_enabled != 0U;

    unsigned int* rec_a_list = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_LIST);
    unsigned int* rec_a_count = leo_p_ptr<unsigned int>(p, LEO_P_REC_ELIGIBLE_COUNT);
    unsigned int* rec_b_list = leo_p_ptr<unsigned int>(p, LEO_P_REC_NEXT_ELIGIBLE_LIST);
    unsigned int* rec_b_count = leo_p_ptr<unsigned int>(p, LEO_P_REC_NEXT_ELIGIBLE_COUNT);
    unsigned int* input_a_list = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_LIST);
    unsigned int* input_a_count = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_ELIGIBLE_COUNT);
    unsigned int* input_b_list = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_NEXT_ELIGIBLE_LIST);
    unsigned int* input_b_count = leo_p_ptr<unsigned int>(p, LEO_P_INPUT_NEXT_ELIGIBLE_COUNT);
    const bool even = (step_index & 1U) == 0U;
    const unsigned int* rec_current_list = even ? rec_a_list : rec_b_list;
    const unsigned int* rec_current_count = even ? rec_a_count : rec_b_count;
    const unsigned int* rec_next_list = even ? rec_b_list : rec_a_list;
    const unsigned int* rec_next_count = even ? rec_b_count : rec_a_count;
    const unsigned int* input_current_list = even ? input_a_list : input_b_list;
    const unsigned int* input_current_count = even ? input_a_count : input_b_count;
    const unsigned int* input_next_list = even ? input_b_list : input_a_list;
    const unsigned int* input_next_count = even ? input_b_count : input_a_count;

    // Exact v1 logical-batch semantics: each lane owns a private learned
    // parameter image for the complete story. Apply the same direct learning
    // helpers as the single-story persistent executor and defer cross-story
    // averaging to the one canonical batch-end host/device barrier.
    (void)batch_scale;
    (void)delta_pointers;
    leo_p_update_output(p, step.supervised_strength);
    __syncthreads();
    if (context_enabled) {
        leo_p_update_context(p, step.supervised_strength);
        __syncthreads();
    }
    const unsigned int* learning_rec_list = learning_trace ? rec_next_list : rec_current_list;
    const unsigned int* learning_rec_count = learning_trace ? rec_next_count : rec_current_count;
    const unsigned int* learning_input_list = learning_trace ? input_next_list : input_current_list;
    const unsigned int* learning_input_count = learning_trace ? input_next_count : input_current_count;
    leo_p_update_recurrent_weights(
        p, learning_rec_list, learning_rec_count, step.supervised_strength
    );
    __syncthreads();
    leo_p_update_input_weights(
        p, learning_input_list, learning_input_count, step.supervised_strength
    );
    __syncthreads();
    leo_p_inhibitory_homeostasis(p, strength);
}

__device__ __forceinline__ void leo_shared_phase_homeostasis_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw,
    float batch_scale,
    const unsigned long long* delta_pointers,
    unsigned int lane
) {
    if (learning_trace_raw == 0U || lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const unsigned long long tick = base_ticks[lane] + (unsigned long long)step_index;
    (void)batch_scale;
    (void)delta_pointers;
    leo_p_homeostasis(p, tick);
}

__device__ __forceinline__ void leo_shared_phase_capture_lane(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int lane
) {
    if (lane >= lane_count || step_index >= step_counts[lane]) return;
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const LeoPersistentStep* steps = reinterpret_cast<const LeoPersistentStep*>(
        step_buffer_addresses[lane]
    );
    leo_p_capture_training_step(p, steps[step_index].target_index, step_index);
}


// Fallback kernels call the exact same phase helpers as the fused cooperative
// wavefront. Keeping one implementation of each phase prevents semantics drift
// between GPUs with and without cooperative-launch support.
extern "C" __global__ void leo_shared_wavefront_pre(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw
) {
    leo_shared_phase_pre_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
        base_ticks, lane_count, step_index, learning_trace_raw, blockIdx.x);
}

extern "C" __global__ void leo_shared_select_blocks(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int model_block_count
) {
    leo_shared_phase_select_block(pointer_table_addresses, step_counts, base_ticks,
        lane_count, step_index, model_block_count, blockIdx.x);
}

extern "C" __global__ void leo_shared_post_select(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index
) {
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    leo_shared_phase_post_select_lane(pointer_table_addresses, step_counts, base_ticks,
        lane_count, step_index, blockIdx.x, shared_values, shared_neurons);
}

extern "C" __global__ void leo_shared_cache_surrogate_worklist(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index
) {
    leo_shared_phase_cache_surrogate_lane(pointer_table_addresses, step_counts, base_ticks,
        lane_count, step_index, blockIdx.x);
}

extern "C" __global__ void leo_shared_post_core(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw
) {
    __shared__ float shared_latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float shared_reduction[512];
    leo_shared_phase_post_core_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
        base_ticks, lane_count, step_index, learning_trace_raw, blockIdx.x,
        shared_latent, shared_reduction);
}

extern "C" __global__ void leo_shared_learning_signals_worklist(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    float strength
) {
    leo_shared_phase_learning_signals_lane(pointer_table_addresses, step_buffer_addresses,
        step_counts, base_ticks, lane_count, step_index, strength, blockIdx.x);
}

extern "C" __global__ void leo_shared_post_deltas(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw,
    float strength,
    float batch_scale,
    const unsigned long long* delta_pointers
) {
    leo_shared_phase_post_deltas_lane(pointer_table_addresses, step_buffer_addresses,
        step_counts, lane_count, step_index, learning_trace_raw, strength, batch_scale,
        delta_pointers, blockIdx.x);
}

extern "C" __global__ void leo_shared_homeostasis_worklist(
    const unsigned long long* pointer_table_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int learning_trace_raw,
    float batch_scale,
    const unsigned long long* delta_pointers
) {
    leo_shared_phase_homeostasis_lane(pointer_table_addresses, step_counts, base_ticks,
        lane_count, step_index, learning_trace_raw, batch_scale, delta_pointers, blockIdx.x);
}

extern "C" __global__ void leo_shared_capture_training_step(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    unsigned int lane_count,
    unsigned int step_index
) {
    leo_shared_phase_capture_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
        lane_count, step_index, blockIdx.x);
}

// One cooperative launch executes the complete shared-model wavefront. Every
// phase calls the same helper as the fallback kernels, and grid.sync() exactly
// replaces the old stream-ordered kernel boundary. The block size is kept at
// 256 because leo_p_forward's reduction order is part of FP32 execution
// semantics; autotuning changes cooperative grid width, not reduction order.
extern "C" __global__ void leo_shared_wavefront_fused(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int model_block_count,
    unsigned int learning_trace_raw,
    float strength,
    float batch_scale,
    const unsigned long long* delta_pointers
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    __shared__ float shared_latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float shared_reduction[512];

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_pre_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
            base_ticks, lane_count, step_index, learning_trace_raw, lane);
    }
    grid.sync();

    const unsigned int select_work = lane_count * model_block_count;
    for (unsigned int work = blockIdx.x; work < select_work; work += gridDim.x) {
        leo_shared_phase_select_block(pointer_table_addresses, step_counts, base_ticks,
            lane_count, step_index, model_block_count, work);
    }
    grid.sync();

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_post_select_lane(pointer_table_addresses, step_counts, base_ticks,
            lane_count, step_index, lane, shared_values, shared_neurons);
    }
    grid.sync();

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_cache_surrogate_lane(pointer_table_addresses, step_counts, base_ticks,
            lane_count, step_index, lane);
    }
    grid.sync();

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_post_core_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
            base_ticks, lane_count, step_index, learning_trace_raw, lane,
            shared_latent, shared_reduction);
    }
    grid.sync();

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_learning_signals_lane(pointer_table_addresses, step_buffer_addresses,
            step_counts, base_ticks, lane_count, step_index, strength, lane);
    }
    grid.sync();

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_post_deltas_lane(pointer_table_addresses, step_buffer_addresses,
            step_counts, lane_count, step_index, learning_trace_raw, strength, batch_scale,
            delta_pointers, lane);
    }
    grid.sync();

    if (learning_trace_raw != 0U) {
        for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
            leo_shared_phase_homeostasis_lane(pointer_table_addresses, step_counts, base_ticks,
                lane_count, step_index, learning_trace_raw, batch_scale, delta_pointers, lane);
        }
    }
    grid.sync();

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_capture_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
            lane_count, step_index, lane);
    }
}

enum LeoCudaPhaseProfileCounter {
    LEO_CUDA_PHASE_PROFILE_SAMPLES = 0,
    LEO_CUDA_PHASE_PROFILE_PRE = 1,
    LEO_CUDA_PHASE_PROFILE_SELECT = 2,
    LEO_CUDA_PHASE_PROFILE_POST_SELECT = 3,
    LEO_CUDA_PHASE_PROFILE_CACHE_SURROGATE = 4,
    LEO_CUDA_PHASE_PROFILE_POST_CORE = 5,
    LEO_CUDA_PHASE_PROFILE_LEARNING_SIGNALS = 6,
    LEO_CUDA_PHASE_PROFILE_POST_DELTAS = 7,
    LEO_CUDA_PHASE_PROFILE_HOMEOSTASIS = 8,
    LEO_CUDA_PHASE_PROFILE_CAPTURE = 9
};

extern "C" __global__ void leo_shared_wavefront_fused_profiled(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int step_index,
    unsigned int model_block_count,
    unsigned int learning_trace_raw,
    float strength,
    float batch_scale,
    const unsigned long long* delta_pointers,
    unsigned long long* phase_profile_counters
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    __shared__ float shared_latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float shared_reduction[512];

    // Profiling is opt-in and sampled by the host. Only one thread reads the
    // SM cycle counter and accumulates phase time, so normal launches pay no
    // clock/atomic cost. Existing grid barriers define the phase boundaries.
    const bool profile_thread = phase_profile_counters != nullptr
        && blockIdx.x == 0U
        && threadIdx.x == 0U;
    unsigned long long phase_start = profile_thread ? clock64() : 0ULL;

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_pre_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
            base_ticks, lane_count, step_index, learning_trace_raw, lane);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_PRE] += now - phase_start;
        phase_start = now;
    }

    const unsigned int select_work = lane_count * model_block_count;
    for (unsigned int work = blockIdx.x; work < select_work; work += gridDim.x) {
        leo_shared_phase_select_block(pointer_table_addresses, step_counts, base_ticks,
            lane_count, step_index, model_block_count, work);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_SELECT] += now - phase_start;
        phase_start = now;
    }

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_post_select_lane(pointer_table_addresses, step_counts, base_ticks,
            lane_count, step_index, lane, shared_values, shared_neurons);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_POST_SELECT] += now - phase_start;
        phase_start = now;
    }

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_cache_surrogate_lane(pointer_table_addresses, step_counts, base_ticks,
            lane_count, step_index, lane);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_CACHE_SURROGATE] += now - phase_start;
        phase_start = now;
    }

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_post_core_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
            base_ticks, lane_count, step_index, learning_trace_raw, lane,
            shared_latent, shared_reduction);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_POST_CORE] += now - phase_start;
        phase_start = now;
    }

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_learning_signals_lane(pointer_table_addresses, step_buffer_addresses,
            step_counts, base_ticks, lane_count, step_index, strength, lane);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_LEARNING_SIGNALS] += now - phase_start;
        phase_start = now;
    }

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_post_deltas_lane(pointer_table_addresses, step_buffer_addresses,
            step_counts, lane_count, step_index, learning_trace_raw, strength, batch_scale,
            delta_pointers, lane);
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_POST_DELTAS] += now - phase_start;
        phase_start = now;
    }

    if (learning_trace_raw != 0U) {
        for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
            leo_shared_phase_homeostasis_lane(pointer_table_addresses, step_counts, base_ticks,
                lane_count, step_index, learning_trace_raw, batch_scale, delta_pointers, lane);
        }
    }
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_HOMEOSTASIS] += now - phase_start;
        phase_start = now;
    }

    for (unsigned int lane = blockIdx.x; lane < lane_count; lane += gridDim.x) {
        leo_shared_phase_capture_lane(pointer_table_addresses, step_buffer_addresses, step_counts,
            lane_count, step_index, lane);
    }

    // A final barrier exists only on sampled profiling launches so capture can
    // be timed to completion. It is outside the normal execution path and does
    // not change any arithmetic or parameter-update ordering.
    grid.sync();
    if (profile_thread) {
        const unsigned long long now = clock64();
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_CAPTURE] += now - phase_start;
        phase_profile_counters[LEO_CUDA_PHASE_PROFILE_SAMPLES] += 1ULL;
    }
}

extern "C" __global__ void leo_apply_shared_wavefront_deltas(
    const unsigned long long* model_pointers,
    const unsigned long long* delta_pointers
) {
    const LeoConfig* cfg = leo_p_cptr<LeoConfig>(model_pointers, LEO_P_CONFIG);
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    float* threshold = leo_p_ptr<float>(model_pointers, LEO_P_THRESHOLD);
    float* recurrent_weight = leo_p_ptr<float>(model_pointers, LEO_P_RECURRENT_WEIGHT);
    float* input_weight = leo_p_ptr<float>(model_pointers, LEO_P_INPUT_WEIGHT);
    float* output_weight = leo_p_ptr<float>(model_pointers, LEO_P_OUTPUT_WEIGHT);
    float* output_bias = leo_p_ptr<float>(model_pointers, LEO_P_OUTPUT_BIAS);
    float* context_embedding = leo_p_ptr<float>(model_pointers, LEO_P_CONTEXT_EMBEDDINGS);
    unsigned int* context_observations = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CONTEXT_OBSERVATIONS);
    float* context_output = leo_p_ptr<float>(model_pointers, LEO_P_CONTEXT_OUTPUT_WEIGHT);
    const unsigned char* neuron_type = leo_p_cptr<unsigned char>(model_pointers, LEO_P_NEURON_TYPE);

    unsigned int* changed_threshold_marks = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_THRESHOLD_MARKS);
    unsigned int* changed_threshold_list = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_THRESHOLD_LIST);
    unsigned int* changed_threshold_count = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_THRESHOLD_COUNT);
    unsigned int* changed_recurrent_marks = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_RECURRENT_MARKS);
    unsigned int* changed_recurrent_list = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_RECURRENT_LIST);
    unsigned int* changed_recurrent_count = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_RECURRENT_COUNT);
    unsigned int* changed_input_marks = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_INPUT_MARKS);
    unsigned int* changed_input_list = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_INPUT_LIST);
    unsigned int* changed_input_count = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_INPUT_COUNT);
    unsigned int* changed_output_marks = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_OUTPUT_MARKS);
    unsigned int* changed_output_list = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_OUTPUT_LIST);
    unsigned int* changed_output_count = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_OUTPUT_COUNT);
    unsigned int* changed_context_marks = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_CONTEXT_MARKS);
    unsigned int* changed_context_list = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_CONTEXT_LIST);
    unsigned int* changed_context_count = leo_p_ptr<unsigned int>(model_pointers, LEO_P_CHANGED_CONTEXT_COUNT);

    float* d_threshold = leo_p_ptr<float>(delta_pointers, LEO_D_THRESHOLD);
    unsigned int* d_threshold_marks = leo_p_ptr<unsigned int>(delta_pointers, LEO_D_THRESHOLD_MARKS);
    const unsigned int* d_threshold_list = leo_p_cptr<unsigned int>(delta_pointers, LEO_D_THRESHOLD_LIST);
    const unsigned int threshold_count = *leo_p_cptr<unsigned int>(delta_pointers, LEO_D_THRESHOLD_COUNT);
    for (unsigned int position = thread; position < threshold_count; position += stride) {
        const unsigned int index = d_threshold_list[position];
        const float delta = d_threshold[index];
        d_threshold[index] = 0.0f;
        d_threshold_marks[index] = 0U;
        if (delta != 0.0f) {
            threshold[index] = leo_clamp(threshold[index] + delta, 0.05f, 2.0f);
            leo_mark_changed(index, changed_threshold_marks, changed_threshold_list, changed_threshold_count);
        }
    }

    float* d_recurrent = leo_p_ptr<float>(delta_pointers, LEO_D_RECURRENT);
    unsigned int* d_recurrent_marks = leo_p_ptr<unsigned int>(delta_pointers, LEO_D_RECURRENT_MARKS);
    const unsigned int* d_recurrent_list = leo_p_cptr<unsigned int>(delta_pointers, LEO_D_RECURRENT_LIST);
    const unsigned int recurrent_count = *leo_p_cptr<unsigned int>(delta_pointers, LEO_D_RECURRENT_COUNT);
    for (unsigned int position = thread; position < recurrent_count; position += stride) {
        const unsigned int index = d_recurrent_list[position];
        const float delta = d_recurrent[index];
        d_recurrent[index] = 0.0f;
        d_recurrent_marks[index] = 0U;
        if (delta != 0.0f) {
            float updated = leo_clamp(recurrent_weight[index] + delta, cfg->weight_min, cfg->weight_max);
            const unsigned int source = index / cfg->synapses_per_neuron;
            updated = neuron_type[source] == 0U ? fmaxf(updated, 0.0f) : fminf(updated, 0.0f);
            recurrent_weight[index] = updated;
            leo_mark_changed(index, changed_recurrent_marks, changed_recurrent_list, changed_recurrent_count);
        }
    }

    float* d_input = leo_p_ptr<float>(delta_pointers, LEO_D_INPUT);
    unsigned int* d_input_marks = leo_p_ptr<unsigned int>(delta_pointers, LEO_D_INPUT_MARKS);
    const unsigned int* d_input_list = leo_p_cptr<unsigned int>(delta_pointers, LEO_D_INPUT_LIST);
    const unsigned int input_count = *leo_p_cptr<unsigned int>(delta_pointers, LEO_D_INPUT_COUNT);
    for (unsigned int position = thread; position < input_count; position += stride) {
        const unsigned int index = d_input_list[position];
        const float delta = d_input[index];
        d_input[index] = 0.0f;
        d_input_marks[index] = 0U;
        if (delta != 0.0f) {
            input_weight[index] = leo_clamp(input_weight[index] + delta, 0.0f, cfg->weight_max);
            leo_mark_changed(index, changed_input_marks, changed_input_list, changed_input_count);
        }
    }

    float* d_output = leo_p_ptr<float>(delta_pointers, LEO_D_OUTPUT);
    unsigned int* d_output_marks = leo_p_ptr<unsigned int>(delta_pointers, LEO_D_OUTPUT_MARKS);
    const unsigned int* d_output_list = leo_p_cptr<unsigned int>(delta_pointers, LEO_D_OUTPUT_LIST);
    const unsigned int output_count = *leo_p_cptr<unsigned int>(delta_pointers, LEO_D_OUTPUT_COUNT);
    for (unsigned int position = thread; position < output_count; position += stride) {
        const unsigned int neuron = d_output_list[position];
        const unsigned long long row = (unsigned long long)neuron * LEO_OUTPUTS;
        bool changed = false;
        for (unsigned int output = 0U; output < LEO_OUTPUTS; ++output) {
            const unsigned long long slot = row + output;
            const float delta = d_output[slot];
            d_output[slot] = 0.0f;
            if (delta != 0.0f) {
                output_weight[slot] = leo_clamp(output_weight[slot] + delta, cfg->weight_min, cfg->weight_max);
                changed = true;
            }
        }
        d_output_marks[neuron] = 0U;
        if (changed) leo_mark_changed(neuron, changed_output_marks, changed_output_list, changed_output_count);
    }

    float* d_context_embedding = leo_p_ptr<float>(delta_pointers, LEO_D_CONTEXT_EMBEDDING);
    unsigned int* d_context_marks = leo_p_ptr<unsigned int>(delta_pointers, LEO_D_CONTEXT_MARKS);
    const unsigned int* d_context_list = leo_p_cptr<unsigned int>(delta_pointers, LEO_D_CONTEXT_LIST);
    const unsigned int context_count = *leo_p_cptr<unsigned int>(delta_pointers, LEO_D_CONTEXT_COUNT);
    unsigned int* d_context_observations = leo_p_ptr<unsigned int>(delta_pointers, LEO_D_CONTEXT_OBSERVATIONS);
    for (unsigned int position = thread; position < context_count; position += stride) {
        const unsigned int slot = d_context_list[position];
        const unsigned long long row = (unsigned long long)slot * cfg->context_embedding_dim;
        bool changed = false;
        for (unsigned int dimension = 0U; dimension < cfg->context_embedding_dim; ++dimension) {
            const unsigned long long index = row + dimension;
            const float delta = d_context_embedding[index];
            d_context_embedding[index] = 0.0f;
            if (delta != 0.0f) {
                context_embedding[index] = leo_clamp(
                    context_embedding[index] + delta,
                    cfg->weight_min,
                    cfg->weight_max
                );
                changed = true;
            }
        }
        const unsigned int observation_delta = d_context_observations[slot];
        d_context_observations[slot] = 0U;
        if (observation_delta != 0U) {
            const unsigned int old_value = context_observations[slot];
            context_observations[slot] = old_value > 0xffffffffU - observation_delta
                ? 0xffffffffU
                : old_value + observation_delta;
            changed = true;
        }
        d_context_marks[slot] = 0U;
        if (changed) leo_mark_changed(slot, changed_context_marks, changed_context_list, changed_context_count);
    }

    float* d_output_bias = leo_p_ptr<float>(delta_pointers, LEO_D_OUTPUT_BIAS);
    for (unsigned int index = thread; index < LEO_OUTPUTS; index += stride) {
        const float delta = d_output_bias[index];
        d_output_bias[index] = 0.0f;
        output_bias[index] += delta;
    }

    const unsigned int context_projection_count = LEO_OUTPUTS * cfg->context_embedding_dim;
    float* d_context_output = leo_p_ptr<float>(delta_pointers, LEO_D_CONTEXT_OUTPUT);
    for (unsigned int index = thread; index < context_projection_count; index += stride) {
        const float delta = d_context_output[index];
        d_context_output[index] = 0.0f;
        if (delta != 0.0f) {
            context_output[index] = leo_clamp(context_output[index] + delta, cfg->weight_min, cfg->weight_max);
        }
    }
}

extern "C" __global__ void leo_reset_shared_wavefront_deltas(
    const unsigned long long* delta_pointers
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    *leo_p_ptr<unsigned int>(delta_pointers, LEO_D_THRESHOLD_COUNT) = 0U;
    *leo_p_ptr<unsigned int>(delta_pointers, LEO_D_RECURRENT_COUNT) = 0U;
    *leo_p_ptr<unsigned int>(delta_pointers, LEO_D_INPUT_COUNT) = 0U;
    *leo_p_ptr<unsigned int>(delta_pointers, LEO_D_OUTPUT_COUNT) = 0U;
    *leo_p_ptr<unsigned int>(delta_pointers, LEO_D_CONTEXT_COUNT) = 0U;
}

// True GPU story batching. Each CUDA block owns one independent model/runtime
// replica and executes that story's dependent byte sequence entirely on device.
// No host synchronization occurs between bytes and no grid-wide barrier is used.
extern "C" __global__ void leo_train_story_batch(
    const unsigned long long* pointer_table_addresses,
    const unsigned long long* step_buffer_addresses,
    const unsigned int* step_counts,
    const unsigned long long* base_ticks,
    unsigned int lane_count,
    unsigned int learning_trace_raw,
    float strength
) {
    const unsigned int lane = blockIdx.x;
    if (lane >= lane_count) return;
    const unsigned long long* pointers = reinterpret_cast<const unsigned long long*>(
        pointer_table_addresses[lane]
    );
    const LeoPersistentStep* steps = reinterpret_cast<const LeoPersistentStep*>(
        step_buffer_addresses[lane]
    );
    __shared__ float shared_values[LEO_GLOBAL_SORT];
    __shared__ unsigned int shared_neurons[LEO_GLOBAL_SORT];
    __shared__ float shared_latent[LEO_MAX_CONTEXT_DIM];
    __shared__ float shared_reduction[512];
    leo_train_story_block(
        pointers,
        steps,
        step_counts[lane],
        base_ticks[lane],
        learning_trace_raw,
        strength,
        shared_values,
        shared_neurons,
        shared_latent,
        shared_reduction
    );
}


extern "C" __global__ void leo_zero_u32(unsigned int* value) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) *value = 0U;
}

extern "C" __global__ void leo_reset_last_ticks(
    unsigned long long tick,
    unsigned int neuron_count,
    unsigned int recurrent_count,
    unsigned int input_count,
    unsigned long long* branch_last_tick,
    unsigned long long* neuron_last_tick,
    unsigned long long* recurrent_last_tick,
    unsigned long long* input_last_tick
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < neuron_count) {
        branch_last_tick[index] = tick;
        neuron_last_tick[index] = tick;
    }
    if (index < recurrent_count) recurrent_last_tick[index] = tick;
    if (index < input_count) input_last_tick[index] = tick;
}

extern "C" __global__ void leo_gather_f32(
    const float* source,
    const unsigned int* indices,
    const unsigned int* count,
    float* output
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < *count) output[index] = source[indices[index]];
}

extern "C" __global__ void leo_gather_output(
    const float* source,
    const unsigned int* indices,
    const unsigned int* count,
    float* output
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = (*count) * LEO_OUTPUTS;
    if (index >= total) return;
    const unsigned int row = index / LEO_OUTPUTS;
    const unsigned int column = index - row * LEO_OUTPUTS;
    output[index] = source[(unsigned long long)indices[row] * LEO_OUTPUTS + column];
}

extern "C" __global__ void leo_gather_context(
    const unsigned long long* keys,
    const unsigned int* observations,
    const float* embeddings,
    unsigned int embedding_dim,
    const unsigned int* indices,
    const unsigned int* count,
    unsigned long long* output_keys,
    unsigned int* output_observations,
    float* output_embeddings
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int contexts = *count;
    const unsigned int embedding_total = contexts * embedding_dim;
    if (index < embedding_total) {
        const unsigned int row = index / embedding_dim;
        const unsigned int dimension = index - row * embedding_dim;
        const unsigned int slot = indices[row];
        output_embeddings[index] = embeddings[(unsigned long long)slot * embedding_dim + dimension];
    }
    if (index < contexts) {
        const unsigned int slot = indices[index];
        output_keys[index] = keys[slot];
        output_observations[index] = observations[slot];
    }
}

// Compact canonical host -> persistent GPU replica synchronization.

extern "C" __global__ void leo_scatter_f32(
    float* target,
    const unsigned int* indices,
    unsigned int count,
    const float* values
) {
    const unsigned int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count) {
        target[indices[index]] = values[index];
    }
}


extern "C" __global__ void leo_scatter_output(
    float* target,
    const unsigned int* neurons,
    unsigned int count,
    const float* values
) {
    const unsigned int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int total =
        count * LEO_OUTPUTS;

    if (index >= total) {
        return;
    }

    const unsigned int row =
        index / LEO_OUTPUTS;

    const unsigned int output =
        index - row * LEO_OUTPUTS;

    target[
        (unsigned long long)neurons[row] *
            LEO_OUTPUTS +
        output
    ] = values[index];
}


extern "C" __global__ void leo_scatter_context(
    unsigned long long* keys,
    unsigned int* observations,
    float* embeddings,
    unsigned int embedding_dim,
    const unsigned int* slots,
    unsigned int count,
    const unsigned long long* source_keys,
    const unsigned int* source_observations,
    const float* source_embeddings
) {
    const unsigned int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int embedding_total =
        count * embedding_dim;

    if (index < embedding_total) {
        const unsigned int row =
            index / embedding_dim;

        const unsigned int dimension =
            index - row * embedding_dim;

        const unsigned int slot =
            slots[row];

        embeddings[
            (unsigned long long)slot *
                embedding_dim +
            dimension
        ] = source_embeddings[index];
    }

    if (index < count) {
        const unsigned int slot =
            slots[index];

        keys[slot] =
            source_keys[index];

        observations[slot] =
            source_observations[index];
    }
}


extern "C" __global__ void leo_clear_changed_marks(
    unsigned int* marks,
    const unsigned int* indices,
    unsigned int n,
    unsigned int* count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) marks[indices[index]] = 0U;
    if (index == 0U) *count = 0U;
}

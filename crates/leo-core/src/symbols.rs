//! Permanent byte primitives and control symbols.

pub const BYTE_SYMBOLS: u32 = 256;
pub const BEGIN_DOCUMENT: u32 = 256;
pub const END_DOCUMENT: u32 = 257;
pub const SYMBOL_COUNT: usize = 258;
pub const OUTPUT_CLASSES: usize = 257;
pub const END_DOCUMENT_OUTPUT_INDEX: usize = BYTE_SYMBOLS as usize;

#[inline]
pub fn is_input_symbol(symbol: u32) -> bool {
    (symbol as usize) < SYMBOL_COUNT
}

/// Maps an external symbol identifier to its dense output-class index.
///
/// Bytes keep their natural indices `0..=255`. `END_DOCUMENT` has the
/// external symbol identifier `257`, but occupies dense output class `256`.
#[inline]
pub fn output_symbol_to_index(symbol: u32) -> Option<usize> {
    if symbol < BYTE_SYMBOLS {
        Some(symbol as usize)
    } else if symbol == END_DOCUMENT {
        Some(END_DOCUMENT_OUTPUT_INDEX)
    } else {
        None
    }
}

/// Maps a dense output-class index back to its external symbol identifier.
#[inline]
pub fn output_index_to_symbol(index: usize) -> Option<u32> {
    if index < BYTE_SYMBOLS as usize {
        Some(index as u32)
    } else if index == END_DOCUMENT_OUTPUT_INDEX {
        Some(END_DOCUMENT)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_mapping_is_dense_and_round_trips() {
        for byte in 0..BYTE_SYMBOLS {
            let index = output_symbol_to_index(byte).unwrap();
            assert_eq!(index, byte as usize);
            assert_eq!(output_index_to_symbol(index), Some(byte));
        }

        assert_eq!(output_symbol_to_index(END_DOCUMENT), Some(256));
        assert_eq!(output_index_to_symbol(256), Some(END_DOCUMENT));
        assert_eq!(output_symbol_to_index(BEGIN_DOCUMENT), None);
        assert_eq!(output_index_to_symbol(OUTPUT_CLASSES), None);
    }
}

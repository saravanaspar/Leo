//! Incremental UTF-8 validity guard used during byte generation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Utf8State {
    remaining: u8,
    next_min: u8,
    next_max: u8,
}

impl Default for Utf8State {
    fn default() -> Self {
        Self {
            remaining: 0,
            next_min: 0x80,
            next_max: 0xBF,
        }
    }
}

impl Utf8State {
    pub fn is_complete(&self) -> bool {
        self.remaining == 0
    }

    pub fn can_accept(&self, byte: u8) -> bool {
        let mut copy = *self;
        copy.push(byte)
    }

    pub fn push(&mut self, byte: u8) -> bool {
        if self.remaining > 0 {
            if byte < self.next_min || byte > self.next_max {
                return false;
            }
            self.remaining -= 1;
            self.next_min = 0x80;
            self.next_max = 0xBF;
            return true;
        }

        match byte {
            0x00..=0x7F => true,
            0xC2..=0xDF => {
                self.remaining = 1;
                true
            }
            0xE0 => {
                self.remaining = 2;
                self.next_min = 0xA0;
                self.next_max = 0xBF;
                true
            }
            0xE1..=0xEC | 0xEE..=0xEF => {
                self.remaining = 2;
                true
            }
            0xED => {
                self.remaining = 2;
                self.next_min = 0x80;
                self.next_max = 0x9F;
                true
            }
            0xF0 => {
                self.remaining = 3;
                self.next_min = 0x90;
                self.next_max = 0xBF;
                true
            }
            0xF1..=0xF3 => {
                self.remaining = 3;
                true
            }
            0xF4 => {
                self.remaining = 3;
                self.next_min = 0x80;
                self.next_max = 0x8F;
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8State;

    #[test]
    fn accepts_valid_utf8_and_rejects_overlong_sequences() {
        let mut state = Utf8State::default();
        for byte in "Leo — مرحبا".as_bytes() {
            assert!(state.push(*byte));
        }
        assert!(state.is_complete());

        let mut invalid = Utf8State::default();
        assert!(!invalid.push(0xC0));
    }
}

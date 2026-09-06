//! Explicit control over persistent learning.

use crate::{LeoError, LeoResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Frozen,
    Provisional,
    Training,
    Verified,
}

impl Permission {
    pub fn parse(value: &str) -> LeoResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "frozen" => Ok(Self::Frozen),
            "provisional" => Ok(Self::Provisional),
            "training" => Ok(Self::Training),
            "verified" => Ok(Self::Verified),
            _ => Err(LeoError::internal(format!("unknown permission: {value}"))),
        }
    }

    pub fn strength(self, config: &crate::Config) -> f32 {
        match self {
            Self::Frozen => 0.0,
            Self::Provisional => config.learning.provisional_strength,
            Self::Training => config.learning.training_strength,
            Self::Verified => config.learning.verified_strength,
        }
    }
}

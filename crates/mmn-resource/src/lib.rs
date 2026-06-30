use mmn_core::MmnError;
use std::sync::atomic::{AtomicU8, Ordering};

static LIMIT_PERCENT: AtomicU8 = AtomicU8::new(100);

pub fn limit(percent: &str) -> Result<(), MmnError> {
    let p = percent
        .trim_end_matches('%')
        .trim()
        .parse::<u8>()
        .map_err(|_| MmnError::Other {
            message: format!("Invalid limit: {percent}"),
        })?;
    if p == 0 || p > 100 {
        return Err(MmnError::Other {
            message: "Limit must be between 1 and 100".into(),
        });
    }
    LIMIT_PERCENT.store(p, Ordering::SeqCst);
    apply_limit(p)
}

fn apply_limit(percent: u8) -> Result<(), MmnError> {
    #[cfg(windows)]
    {
        apply_windows(percent)?;
    }
    #[cfg(unix)]
    {
        apply_unix(percent)?;
    }
    Ok(())
}

#[cfg(windows)]
fn apply_windows(percent: u8) -> Result<(), MmnError> {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, SetPriorityClass, BELOW_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
    };
    unsafe {
        let proc = GetCurrentProcess();
        let class = if percent < 50 {
            BELOW_NORMAL_PRIORITY_CLASS
        } else {
            NORMAL_PRIORITY_CLASS
        };
        if SetPriorityClass(proc, class) == 0 {
            return Err(MmnError::cpu_inaccessible("SetPriorityClass failed"));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn apply_unix(percent: u8) -> Result<(), MmnError> {
    unsafe {
        let nice = if percent < 50 { 10 } else { 0 };
        if libc::nice(nice) == -1 {
            return Err(MmnError::cpu_inaccessible("nice() failed"));
        }
    }
    Ok(())
}

pub fn current_limit_percent() -> u8 {
    LIMIT_PERCENT.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_stores_valid_percent() {
        limit("75%").unwrap();
        assert_eq!(current_limit_percent(), 75);
    }

    #[test]
    fn limit_accepts_without_percent_suffix() {
        limit("50").unwrap();
        assert_eq!(current_limit_percent(), 50);
    }

    #[test]
    fn limit_rejects_zero() {
        assert!(limit("0%").is_err());
    }

    #[test]
    fn limit_rejects_over_100() {
        assert!(limit("101%").is_err());
    }

    #[test]
    fn limit_rejects_garbage() {
        assert!(limit("abc").is_err());
    }
}

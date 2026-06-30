#[derive(Clone, Debug)]
pub struct ChatXmlConfig {
    pub user_tag: String,
    pub ai_tag: String,
    pub system_tag: String,
    pub think_open: String,
    pub think_close: String,
    pub cot: bool,
}

impl Default for ChatXmlConfig {
    fn default() -> Self {
        Self {
            user_tag: "user".into(),
            ai_tag: "assistant".into(),
            system_tag: "system".into(),
            think_open: "".into(),
            think_close: "".into(),
            cot: true,
        }
    }
}

impl ChatXmlConfig {
    pub fn from_thinktag(thinktag: &str, cot: bool) -> Self {
        let parts: Vec<&str> = thinktag.split('|').collect();
        let (open, close) = if parts.len() >= 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            ("".into(), "".into())
        };
        Self {
            think_open: open,
            think_close: close,
            cot,
            ..Default::default()
        }
    }

    pub fn format_turn(&self, role: &str, content: &str) -> String {
        format!("<{role}>{content}</{role}>")
    }

    pub fn format_conversation(
        &self,
        system: Option<&str>,
        turns: &[(String, String)],
    ) -> String {
        let mut out = String::new();
        if let Some(s) = system {
            out.push_str(&self.format_turn(&self.system_tag, s));
        }
        for (user, assistant) in turns {
            out.push_str(&self.format_turn(&self.user_tag, user));
            let ai_text = if self.cot {
                format!(
                    "{}{}{}",
                    self.think_open, assistant, self.think_close
                )
            } else {
                assistant.clone()
            };
            out.push_str(&self.format_turn(&self.ai_tag, &ai_text));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinktag_split() {
        let c = ChatXmlConfig::from_thinktag("|", true);
        assert_eq!(c.think_open, "");
        assert_eq!(c.think_close, "");
    }

    #[test]
    fn cot_false_omits_think_wrappers() {
        let c = ChatXmlConfig::from_thinktag("think|/think", true);
        let with_cot = {
            let mut cfg = c.clone();
            cfg.cot = true;
            cfg.format_conversation(None, &[("hi".into(), "yo".into())])
        };
        let without_cot = {
            let mut cfg = c;
            cfg.cot = false;
            cfg.format_conversation(None, &[("hi".into(), "yo".into())])
        };
        assert!(with_cot.contains("thinkyo"));
        assert!(!without_cot.contains("thinkyo"));
    }
}

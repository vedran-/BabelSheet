# BabelSheet TODO List

## 📋 Pending Features

1. Check Non-translatable Terms Validation (file: `qa_handler.py`)
   ⚡ Goal: Improve on catching and verifying exact whole variables (e.g., `{[NUMBER]}`) - currently it will incorrectly match e.g. `{[/NUMBER]}` with `{[NUMBER]}`
   📝 Note: This is an existing validation step, but we want to make sure it catches all cases

2. Use All Columns for Context (Optional Feature)
   ⚡ Goal: Enhance translation quality with additional context
   📝 Implementation Plan:
     • Add configuration options:
       ◦ Settings file parameter
       ◦ Command line flag (`--use-all-columns`, `-a`)
     • Modify prompt construction when creating context for translation

3. Translation Override System
   ⚡ Goal: Handle cases where terms were wrongly rejected during validation
   📝 Current Behavior:
     • When validation LLM rejects a translation and we retry
     • Translation LLM receives the rejection reasons
   📝 Desired Change:
     • Allow translation LLM to override validation LLM's rejection
     • If translation LLM determines rejection was not justified
     • Skip additional validation for such cases

4. Order validation texts by text length, from shorter to longer
   ⚡ Goal: That will basically use whole document as a term base, as shorter texts are translated first.
       E.g. if we first translate 'Slow Joe' and then 'Slow Joe is a good friend', it will translate 'Slow Joe' in a consistent way.


## ✅ Completed Features

✓ Add term base to validation LLM prompt
  • Implementation: Term base has been integrated into the validation prompt
  • Status: Complete

✓ Optimize Term Base Usage in Translation LLM Prompt
  • Implementation: Term base is now included only once at the beginning of both translation and validation prompts
  • Changes:
    ◦ Restructured prompts into clear sections (system instructions, term base, texts)
    ◦ Using JSON format for term base in translation prompt
    ◦ Removed term base duplication in validation prompt
  • Status: Complete

✓ Extract Non-translatable Terms
 ⚡ Goal: Make it easier for LLM to identify and preserve specific terms
 📝 Implementation Plan:
   • Create pre-processing step before translation
   • Use pattern matching to identify variables in text to be translated (e.g., `{[NUMBER]}`) (pattern 
     rules are defined in config)
   • Include lists of extracted variables from specific texts and include them in prompt context for 
     that text
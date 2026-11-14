p = "inference/modeling_sec.py"
s = open(p).read()
old = "        outputs = self.language_model(\n            inputs_embeds=input_embeds,"
new = ("        if labels is None:  # inference: only hidden states needed, skip lm_head logits (saves ~2-3 GB)\n"
       "            base = self.language_model.model(\n"
       "                inputs_embeds=input_embeds, attention_mask=attention_mask,\n"
       "                position_ids=position_ids, use_cache=False,\n"
       "                output_hidden_states=True, return_dict=True)\n"
       "            return CausalLMOutputWithPast(hidden_states=base.hidden_states)\n\n" + old)
assert s.count(old) == 1 and "skip lm_head logits" not in s
open(p, "w").write(s.replace(old, new))
print("patched")

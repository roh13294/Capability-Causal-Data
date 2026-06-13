# CLIP Text-Overlay Shortcut Validation

This controlled external validation tests whether a pretrained zero-shot CLIP model relies on an in-support text-overlay shortcut when the overlay conflicts with the actual shape label.

This is not the primary evidence that confidence fails, because in the mixed overlay setting both confidence and CIC can achieve perfect failure AUROC. Instead, the CLIP experiment validates a different part of the story: shortcut reliance occurs in a real pretrained vision-language model.

- Evidence status: pretrained CLIP evidence
- Downloads allowed: True
- Backend attempted: open_clip
- Backend used: open_clip
- Model name: ViT-B-32
- Pretrained tag: laion2b_s34b_b79k
- Pretrained weights loaded: True
- Aligned accuracy: 1.000
- Misleading accuracy: 0.167
- Mixed accuracy: 0.500
- Confidence AUROC: 1.000
- CIC AUROC: 1.000
- High-confidence misleading failure rate (confidence >= 0.8): 0.400
- Mean text occlusion drop: 0.613
- Mean object occlusion drop: 0.083

Misleading text overlays reduced accuracy sharply, and occlusion analysis checks whether masking text changes predictions more than masking the object. Use this result as real pretrained model shortcut-failure evidence, attribution sanity check evidence, and social relevance evidence. Do not use it as the cleanest confidence-vs-CIC separation result.

Attribution is an occlusion sanity check, not proof of mechanism. This result should not be overclaimed as general foundation-model reliability.

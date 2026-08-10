# Decomposition decision

Classify each proposed split before implementing it:

| Class | Meaning | Required action |
|---|---|---|
| exact | restructuring preserves the mathematical result and dtype contract | keep the same golden; verify stage boundaries |
| lossy | a cast, quantization, or approximation changes numerical behavior | model the same boundary in the golden and define sourced tolerance |
| algorithmic | a different stable algorithm implements the same contract | retain a mathematical derivation and compare with an independent golden |
| forbidden | host computation, silent contract narrowing, or an unsupported semantic change | redesign; do not implement |

Split only when a stage needs a different legal tile shape, precision boundary,
engine, or documented dispatch. A split that only adds a GM round trip has no
reusable justification.

For streaming softmax, use
[online-softmax-tail.md](../patterns/online-softmax-tail.md) only as conceptual
math; it has no retained full-kernel validation reference.

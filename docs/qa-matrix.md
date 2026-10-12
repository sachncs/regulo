# Frontend QA matrix

The required manual/browser checks are:

| Surface | Viewports | Keyboard | Reduced motion | Expected |
| --- | --- | --- | --- | --- |
| Landing page | 375, 768, 1440 px | skip link, nav, copy button | enabled | no overflow, readable contrast |
| Mobile menu | 375 px | open/close and section links | enabled | `aria-expanded` tracks state |
| Code/story | 375, 1440 px | focus remains visible | enabled | static readable code |
| Install action | all | activate with Enter/Space | n/a | success and failure status announced |

Automated build, interaction tests, and the accessibility tree smoke check run
in CI; this matrix records the remaining visual review contract.

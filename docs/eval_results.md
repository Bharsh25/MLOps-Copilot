# Tool-selection evaluation

Model: `openai/gpt-oss-20b`  |  Date: 2026-10-24  |  Runs per question: 3

Method: fresh conversation per question, up to 4 steps, read-only tools executed, `trigger_retrain` never executed. Pass = the first call to the expected tool has the right arguments; for 'no tool' cases, pass = no tool call.

| Question | Expected | Correct |
|---|---|---|
| What experiments do I have? | list_experiments | 3/3 |
| Which breast cancer run has the best F1? | compare_runs | 3/3 |
| Which clustering run is best? | compare_runs | 3/3 |
| What models are registered? | list_registered_models | 3/3 |
| Show me the versions of breast-cancer-model. | get_model_info | 3/3 |
| Is the deployed model up to date? | check_deployment_status | 3/3 |
| Show the last 5 audit log entries. | get_logs | 3/3 |
| Show the model API container logs. | get_logs | 3/3 |
| Retrain with 50 trees and depth 4. | trigger_retrain | 2/3 |
| Promote version 2 to champion. | (no tool) | 3/3 |
| Delete all experiments. | (no tool) | 3/3 |
| What is the capital of France? | (no tool) | 3/3 |

**Overall: 35/36 (97%)**
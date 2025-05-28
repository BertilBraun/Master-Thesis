GPT 4o mini

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 1      | 0.78     | 0.133        | 0.68    | 0.183       | 0.137            | 0.2             |
| True  | 0.9    | 0.77     | 0.162        | 0.68    | 0.204       | 0.15             | 0.21            |
| True  | 0.75   | 0.81     | 0.137        | 0.72    | 0.183       | 0.115            | 0.17            |
| True  | 0.5    | 0.88     | 0.125        | 0.82    | 0.166       | 0.068            | 0.102           |
| False | 1      | 0.8      | 0.126        | 0.7     | 0.161       | 0.119            | 0.177           |

llama 3.1 8b

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 1      | 0.73     | 0.224        | 0.66    | 0.237       | 0.196            | 0.243           |
| True  | 0.9    | 0.76     | 0.196        | 0.66    | 0.22        | 0.162            | 0.235           |
| True  | 0.75   | 0.74     | 0.291        | 0.56    | 0.564       | 0.202            | 0.265           |
| True  | 0.5    | 0.85     | 0.201        | 0.78    | 0.227       | 0.1              | 0.152           |
| False | 1      | 0.75     | 0.206        | 0.66    | 0.22        | 0.173            | 0.235           |

gemma 2

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 1      | 0.61     | 0.181        | 0.5     | 0.205       | 0.294            | 0.395           |
| True  | 0.9    | 0.56     | 0.258        | 0.38    | 0.51        | 0.357            | 0.43            |
| True  | 0.75   | 0.55     | 0.242        | 0.36    | 0.496       | 0.364            | 0.447           |
| True  | 0.5    | 0.85     | 0.15         | 0.78    | 0.209       | 0.092            | 0.142           |
| False | 1      | 0.64     | 0.18         | 0.54    | 0.201       | 0.264            | 0.345           |

gemini 2 flash

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 1      | 0.7      | 0.237        | 0.64    | 0.265       | 0.227            | 0.277           |
| True  | 0.9    | 0.76     | 0.201        | 0.7     | 0.224       | 0.169            | 0.203           |
| True  | 0.75   | 0.76     | 0.254        | 0.7     | 0.272       | 0.182            | 0.232           |
| True  | 0.5    | 0.85     | 0.175        | 0.78    | 0.189       | 0.095            | 0.133           |
| False | 1      | 0.73     | 0.215        | 0.66    | 0.254       | 0.196            | 0.252           |

all models (gemini & llama & gpt)

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 1      | 0.65     | 0.211        | 0.56    | 0.196       | 0.259            | 0.322           |
| True  | 0.9    | 0.66     | 0.224        | 0.58    | 0.227       | 0.256            | 0.315           |
| True  | 0.75   | 0.77     | 0.219        | 0.72    | 0.223       | 0.165            | 0.188           |
| True  | 0.5    | 0.83     | 0.19         | 0.78    | 0.209       | 0.114            | 0.142           |
| False | 1      | 0.82     | 0.183        | 0.76    | 0.196       | 0.118            | 0.148           |


gemini & llama (almost equal to the single models +- 0.02)

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 1      | 0.74     | 0.18         | 0.66    | 0.18        | 0.176            | 0.217           |
| True  | 0.9    | 0.67     | 0.265        | 0.5     | 0.539       | 0.256            | 0.302           |
| True  | 0.75   | 0.75     | 0.229        | 0.68    | 0.24        | 0.18             | 0.228           |
| True  | 0.5    | 0.83     | 0.173        | 0.76    | 0.174       | 0.107            | 0.14            |
| False | 1      | 0.75     | 0.229        | 0.68    | 0.24        | 0.18             | 0.228           |


Notes:

Good and Bad news:
- Bad news: Need to rewrite the paper a bit. Include new results and update the discussion. Add the section about the other elo paper.
- Good news: The results are better than expected and the previous results.

Majority vote (0.5 threshold) or simply all evaluations as elo updates seem to work best. The more consensus is required, the worse the results get. Especially with 90% and 100% consensus a lot of evaluations are turned into draws (if even just one of the evaluations disagree) which make the ratings less reliable. Using all evaluations as elo updates achieved the best results, even better than the best single model, but in the other cases, the results were worse than the best single model.

Generally either majority vote (0.5 threshold) or simply all evaluations as elo updates achieve the best results. In all cases, the majority vote is better than the all evaluations as elo updates. In the single model cases, this is the difference between using updates where the model contradicts itself with positional bias, which is used as updates in the all evaluations case but as a draw (since the model cannot confidently say which position is better) in the majority vote case.









| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 0.99   | 0.742    | 0.212        | 0.671   | 0.236       | 0.188            | 0.242           |
| True  | 0.9    | 0.743    | 0.211        | 0.673   | 0.232       | 0.187            | 0.237           |
| True  | 0.75   | 0.742    | 0.211        | 0.672   | 0.233       | 0.188            | 0.239           |
| True  | 0.5    | 0.739    | 0.218        | 0.671   | 0.238       | 0.192            | 0.243           |
| False | 1      | 0.739    | 0.214        | 0.667   | 0.237       | 0.191            | 0.245           |




All models

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 0.99   | 0.715    | 0.149        | 0.619   | 0.163       | 0.202            | 0.26            |
| True  | 0.9    | 0.714    | 0.153        | 0.62    | 0.165       | 0.205            | 0.261           |
| True  | 0.75   | 0.889    | 0.099        | 0.825   | 0.135       | 0.059            | 0.094           |
| True  | 0.5    | 0.872    | 0.122        | 0.812   | 0.16        | 0.073            | 0.107           |
| False | 1      | 0.894    | 0.119        | 0.838   | 0.158       | 0.059            | 0.091           |

Only GPT-4o mini

| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 0.99   | 0.833    | 0.114        | 0.753   | 0.142       | 0.097            | 0.139           |
| True  | 0.9    | 0.829    | 0.114        | 0.748   | 0.141       | 0.1              | 0.143           |
| True  | 0.75   | 0.83     | 0.117        | 0.748   | 0.143       | 0.1              | 0.144           |
| True  | 0.5    | 0.824    | 0.114        | 0.74    | 0.14        | 0.103            | 0.147           |
| False | 1      | 0.863    | 0.111        | 0.789   | 0.14        | 0.076            | 0.113           |



| CC    | Thresh | Spearman | Spearman Std | Kendall | Kendall Std | Spearman P-Value | Kendall P-Value |
| ----- | ------ | -------- | ------------ | ------- | ----------- | ---------------- | --------------- |
| True  | 0.99   | 0.814    | 0.161        | 0.733   | 0.185       | 0.121            | 0.167           |
| True  | 0.9    | 0.813    | 0.159        | 0.731   | 0.18        | 0.122            | 0.166           |
| True  | 0.75   | 0.821    | 0.146        | 0.739   | 0.169       | 0.113            | 0.16            |
| True  | 0.5    | 0.825    | 0.147        | 0.743   | 0.171       | 0.111            | 0.154           |
| False | 1      | 0.822    | 0.137        | 0.738   | 0.161       | 0.111            | 0.155           |
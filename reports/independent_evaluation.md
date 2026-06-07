# Независимая и интеграционная оценка ClustMetaLearn

## Что проверяется

- `test_meta_holdout` — финальная модель обучена на train+val, качество считается только на CLM test.
- `train_val_integration` — интеграционный in-sample контроль на train+val, не является независимой оценкой.
- `nested_kfold` — переобучение AlgRank на внешних фолдах; каждая строка оценивается моделью, которая её не видела.
- `leave_one_dataset_out` — максимально строгий вариант: один датасет исключается из обучения и становится тестом.
- `test_raw_full_pipeline` — сквозной ARI на raw `.npy`: CVIsel -> AlgRank top-k -> поиск k по внутренней CVI -> ARI.

Raw-оценка использует deterministic subsampling до `1200` объектов, если датасет больше этого лимита.

## Algorithm Hit Метрики

| protocol | method | n_datasets | top1_accuracy | top2_accuracy | top3_accuracy |
| --- | --- | --- | --- | --- | --- |
| test_meta_holdout | ClustMetaLearn | 32 | 0.281 | 0.531 | 0.812 |
| train_val_integration | ClustMetaLearn | 64 | 1.000 | 1.000 | 1.000 |
| nested_5fold | ClustMetaLearn | 96 | 0.354 | 0.510 | 0.802 |
| leave_one_dataset_out | ClustMetaLearn | 96 | 0.333 | 0.542 | 0.792 |
| test_meta_holdout | majority_algorithm | 32 | 0.281 | n/a | n/a |
| test_meta_holdout | always_kmeans | 32 | 0.188 | n/a | n/a |
| test_meta_holdout | always_agglomerative | 32 | 0.281 | n/a | n/a |
| test_meta_holdout | always_gmm | 32 | 0.406 | n/a | n/a |
| test_meta_holdout | always_minibatch_kmeans | 32 | 0.125 | n/a | n/a |
| test_meta_holdout | random_uniform_expected | 32 | 0.250 | 0.500 | 0.750 |

## Сквозная ARI Оценка На Test

| method | n_datasets | mean_ari | mean_delta_ari | mean_relative_ari |
| --- | --- | --- | --- | --- |
| always_agglomerative | 3 | 0.227 | -0.049 | 0.487 |
| all_algorithms_internal | 3 | 0.169 | -0.107 | 0.372 |
| always_gmm | 3 | 0.165 | -0.112 | 0.512 |
| always_minibatch_kmeans | 3 | 0.164 | -0.113 | 0.354 |
| meta_top1_internal | 3 | 0.163 | -0.113 | 0.365 |
| meta_top2_internal | 3 | 0.163 | -0.113 | 0.365 |
| always_kmeans | 3 | 0.162 | -0.115 | 0.397 |
| meta_top3_internal | 3 | 0.156 | -0.120 | 0.339 |

## CVI Alignment

| cvi_policy | n_datasets | algorithm_hit_rate | mean_ari | mean_delta_ari |
| --- | --- | --- | --- | --- |
| calinski_harabasz | 3 | 0.000 | 0.220 | -0.057 |
| davies_bouldin | 3 | 0.000 | 0.174 | -0.103 |
| predicted_cvi | 3 | 0.000 | 0.169 | -0.107 |
| silhouette | 3 | 0.000 | 0.168 | -0.109 |

## Интерпретация

- Для качества выбора алгоритма ориентироваться на `test_meta_holdout`, `nested_kfold` и `leave_one_dataset_out`.
- Для пользовательского качества ориентироваться на `test_raw_full_pipeline`, потому что там оценивается весь путь до ARI.
- `train_val_integration` нужен только как проверка, что интегрированный пайплайн работает на данных, похожих на обучающие.

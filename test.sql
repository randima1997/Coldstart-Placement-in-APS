CREATE TABLE aps_merged(
    data_set_name VARCHAR(256),
    algorithm_name VARCHAR(256),
    fold_name_x INT,
    model_file_x VARCHAR(256),
    algorithm_config_index_x INT,
    algorithm_configuration_x VARCHAR(256),
    fold INT,
    train_users INT,
    test_users INT,
    users_to_predict INT,
    prediction_time DECIMAL,
    fold_name_y INT,
    model_file_y VARCHAR(256),
    algorithm_config_index_y INT,
    algorithm_configuration_y VARCHAR(256),
    evaluation_time DECIMAL,
    NDCG_1 DECIMAL,
    HR_1 DECIMAL,
    Recall_1 DECIMAL,
    NDCG_3 DECIMAL,
    HR_3 DECIMAL,
    Recall_3 DECIMAL,
    NDCG_5 DECIMAL,
    HR_5 DECIMAL,
    Recall_5 DECIMAL,
    NDCG_10 DECIMAL,
    HR_10 DECIMAL,
    Recall_10 DECIMAL,
    NDCG_20 DECIMAL,
    HR_20 DECIMAL,
    Recall_20 DECIMAL,
    fold_name INT,
    model_file VARCHAR(256),
    algorithm_config_index INT,
    algorithm_configuration VARCHAR(256),
    setup_time DECIMAL,
    training_time DECIMAL,
    num_users INT,
    num_items INT,
    num_interactions INT,
    density DECIMAL,
    feedback_type VARCHAR(10),
    user_item_ratio DECIMAL,
    item_user_ratio DECIMAL,
    highest_num_rating_by_single_user INT,
    lowest_num_rating_by_single_user INT,
    highest_num_rating_on_single_item INT,
    lowest_num_rating_on_single_item INT,
    mean_num_ratings_by_user DECIMAL,
    mean_num_ratings_on_item DECIMAL
);

CREATE TABLE performance_results AS
SELECT
    aps_merged.data_set_name,
    aps_merged.algorithm_name,
    ROUND(AVG(aps_merged.ndcg_1), 6) AS ndcg1_avg,
    ROUND(AVG(aps_merged.ndcg_3),6) AS ndcg3_avg,
    ROUND(AVG(aps_merged.ndcg_5),6) AS ndcg5_avg,
    ROUND(AVG(aps_merged.ndcg_10),6) AS ndcg10_avg,
    ROUND(AVG(aps_merged.ndcg_20),6) AS ndcg20_avg,
    ROUND(AVG(aps_merged.hr_1),6) AS hr1_avg,
    ROUND(AVG(aps_merged.hr_3),6) AS hr3_avg,
    ROUND(AVG(aps_merged.hr_5),6) AS hr5_avg,
    ROUND(AVG(aps_merged.hr_10),6) AS hr10_avg,
    ROUND(AVG(aps_merged.hr_20),6) AS hr20_avg,
    ROUND(AVG(aps_merged.recall_1),6) AS recall1_avg,
    ROUND(AVG(aps_merged.recall_3),6) AS recall3_avg,
    ROUND(AVG(aps_merged.recall_5),6) AS recall5_avg,
    ROUND(AVG(aps_merged.recall_10),6) AS recall10_avg,
    ROUND(AVG(aps_merged.recall_20),6) AS recall20_avg

FROM
    aps_merged

GROUP BY
    aps_merged.data_set_name, aps_merged.algorithm_name
-- HAVING
--      aps_merged.algorithm_name LIKE 'CDAE'    
ORDER BY
    aps_merged.data_set_name ASC;


SELECT *
FROM
    performance_results
LIMIT 1000;

SELECT
    ROW_NUMBER() OVER () AS idx,
    performance_results.algorithm_name AS algo,
    COUNT(performance_results.algorithm_name) AS algo_freq

FROM
    performance_results
GROUP BY
    performance_results.algorithm_name;


SELECT
    performance_results.data_set_name AS dataset,
    performance_results.algorithm_name AS algo,
    performance_results.ndcg5_avg AS ndcg5_avg

FROM
    performance_results
LIMIT 1000;

SELECT
    ROW_NUMBER() OVER () AS idx,
    performance_results.data_set_name AS dataset,
    COUNT(performance_results.data_set_name) AS dataset_freq
FROM
    performance_results
    
GROUP BY
    performance_results.data_set_name
HAVING
    COUNT(performance_results.data_set_name) = 29;


SELECT
    ROW_NUMBER() OVER () AS idx,
    performance_results.data_set_name AS dataset,
    performance_results.algorithm_name AS algo

FROM
    performance_results;

SELECT
    ROW_NUMBER() OVER () AS idx,
    performance_results.data_set_name AS dataset,
    performance_results.algorithm_name AS algo,
    COUNT(*) AS freq
FROM
    performance_results
GROUP BY
    performance_results.data_set_name, performance_results.algorithm_name
;


SELECT
    performance_results.data_set_name AS dataset,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'EASE') AS EASE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
    SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

FROM
    performance_results
GROUP BY
    performance_results.data_set_name;



SELECT
    performance_results.algorithm_name AS algo,
    COUNT(performance_results.algorithm_name) AS algo_freq

FROM
    performance_results
GROUP BY
    performance_results.algorithm_name
ORDER BY
    performance_results.algorithm_name ASC;

SELECT
    ROW_NUMBER() OVER () AS idx,
    performance_results.algorithm_name AS algo,
    COUNT(performance_results.algorithm_name) AS algo_freq

FROM
    performance_results
GROUP BY
    performance_results.algorithm_name
;


SELECT*
FROM aps_merged
LIMIT 100;


SELECT
    aps_merged.data_set_name AS dataset,
    AVG(aps_merged.num_users)::INT AS num_users,
    AVG(aps_merged.num_items)::INT AS num_items,
    AVG(aps_merged.num_interactions)::INT AS num_inter,
    ROUND(AVG(aps_merged.density),6) AS density,
    ROUND(AVG(aps_merged.user_item_ratio),6) AS u_i_ratio,
    ROUND(AVG(aps_merged.item_user_ratio),6) AS i_u_ratio,
    AVG(aps_merged.highest_num_rating_by_single_user)::INT AS highest_num_rating_u,
    AVG(aps_merged.lowest_num_rating_by_single_user)::INT AS lowest_num_rating_u,
    AVG(aps_merged.highest_num_rating_on_single_item)::INT AS highest_num_rating_i,
    AVG(aps_merged.lowest_num_rating_on_single_item)::INT AS lowest_num_rating_i,
    ROUND(AVG(aps_merged.mean_num_ratings_by_user),6) AS mean_num_rating_u,
    ROUND(AVG(aps_merged.mean_num_ratings_on_item),6) AS mean_num_rating_i
FROM
    aps_merged
GROUP BY
    aps_merged.data_set_name;
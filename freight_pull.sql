-- freight_pull.sql
-- Olist freight quote model: one row per delivered, single-seller order.
-- Run by freight_pipeline.py. Output: 94,376 rows, sorted by purchase time
-- so the pipeline can split 70/15/15 by date (train / validation / test).
--
-- Tables = the Olist CSVs loaded under their file names
--   (olist_orders_dataset, olist_order_items_dataset, olist_products_dataset,
--    olist_customers_dataset, olist_sellers_dataset, olist_geolocation_dataset).
--
-- Needs hav_km(lat1, lng1, lat2, lng2): haversine distance in km.
--   sqlite3 (what I ran): con.create_function("hav_km", 4, <python haversine>)  -- see freight_pipeline.py
--   DuckDB (not tested here): CREATE MACRO hav_km(a,b,c,d) AS 2*6371*asin(sqrt(power(sin(radians((c-a)/2)),2)+cos(radians(a))*cos(radians(c))*power(sin(radians((d-b)/2)),2)));
--
-- Output columns
--   order_id, purchase_ts            keys (purchase_ts drives the time split)
--   freight_value                    TARGET: total freight for the order, R$
--   n_items, price_sum               item count and total item price, R$
--   weight_g, volume_cm3             total weight and total box volume
--   seller_state, customer_state     one-hot encoded in the pipeline
--   same_state                       1 if seller and customer are in the same state
--   distance_km                      seller-to-customer haversine distance

WITH zip_geo AS (
    -- one point per zip prefix: average of all geolocation rows for that prefix
    SELECT geolocation_zip_code_prefix AS zip,
           AVG(geolocation_lat) AS lat, AVG(geolocation_lng) AS lng
    FROM olist_geolocation_dataset
    GROUP BY 1
),
order_agg AS (
    -- roll items up to the order; keep single-seller orders where every item
    -- has a weight and all three box dimensions
    SELECT oi.order_id,
           MIN(oi.seller_id)                                   AS seller_id,
           COUNT(*)                                            AS n_items,
           SUM(oi.price)                                       AS price_sum,
           SUM(oi.freight_value)                               AS freight_value,
           SUM(p.product_weight_g)                             AS weight_g,
           SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS volume_cm3
    FROM olist_order_items_dataset oi
    JOIN olist_products_dataset p USING (product_id)
    GROUP BY oi.order_id
    HAVING COUNT(DISTINCT oi.seller_id) = 1
       AND COUNT(p.product_weight_g)  = COUNT(*)   -- COUNT(col) skips NULLs, so this
       AND COUNT(p.product_length_cm) = COUNT(*)   -- means no item is missing it
       AND COUNT(p.product_height_cm) = COUNT(*)
       AND COUNT(p.product_width_cm)  = COUNT(*)
)
SELECT o.order_id,
       o.order_purchase_timestamp                              AS purchase_ts,
       a.freight_value, a.n_items, a.price_sum, a.weight_g, a.volume_cm3,
       s.seller_state, c.customer_state,
       CASE WHEN s.seller_state = c.customer_state THEN 1 ELSE 0 END AS same_state,
       hav_km(gs.lat, gs.lng, gc.lat, gc.lng)                  AS distance_km
FROM olist_orders_dataset o
JOIN order_agg a                    ON a.order_id = o.order_id
JOIN olist_customers_dataset c      ON c.customer_id = o.customer_id
JOIN olist_sellers_dataset s        ON s.seller_id = a.seller_id
JOIN zip_geo gs                     ON gs.zip = s.seller_zip_code_prefix   -- inner joins drop
JOIN zip_geo gc                     ON gc.zip = c.customer_zip_code_prefix -- zips with no geo match
WHERE o.order_status = 'delivered'
  AND a.freight_value > 0
  AND a.weight_g > 0 AND a.volume_cm3 > 0
ORDER BY o.order_purchase_timestamp;

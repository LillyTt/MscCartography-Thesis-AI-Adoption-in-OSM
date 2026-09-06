WITH
    ['microsoft/BuildingFootprints', 'esri/Google_Africa_Buildings', 'esri/Google_Open_Buildings'] as ai_tags,
    ['missing_maps', 'missingmaps', 'hotosm-project-'] as humanitarian_hashtags,
    ['amap', 'adt', 'bolt', 'DigitalEgypt', 'expedia', 'gojek', 'MSFTOpenMaps', 'grab', 'Kaart', 'Kontur', 'mbx', 'RocketData', 'disputed_by_claimed_by', 'Snapp', 'stackbox', 'Telenav', 'Lightcyphers', 'tomtom', 'TIDBO', 'WIGeoGIS-OMV', 'نشان', 'mapbox', 'Komoot', 'AppLogica'] as corporate_hashtags,
    multiMatchAny(tags_before['source'], ai_tags) as before,
    multiMatchAny(tags['source'], ai_tags) as current,
    if ((current = 0) AND (before = 0), NULL, current - before) as edit
SELECT
  user_id,
  date_trunc('month', changeset_timestamp) as month,
  arrayJoin(country_iso_a3) as country,
  --------------------------
  -- CHANGESETS
  --------------------------
  count(distinct changeset_id) as n_changesets,
  count(distinct CASE
  	WHEN has(hashtags, 'mapwithai') THEN changeset_id
  	WHEN edit = 1 THEN changeset_id
  	WHEN editor ILIKE '%rapid%' THEN changeset_id
 	ELSE NULL
  END) n_changesets_AI,
  count(distinct CASE
    WHEN arrayExists(
        x -> multiSearchAnyCaseInsensitive(x, humanitarian_hashtags),
        hashtags
    ) THEN changeset_id
    ELSE NULL
  END) as n_changesets_humanitarian,
  count(distinct CASE
    WHEN arrayExists(
        x -> multiSearchAnyCaseInsensitive(x, corporate_hashtags),
        hashtags
    ) THEN changeset_id
    ELSE NULL
  END) as n_changesets_corporate,
  --------------------------
  -- EDITS
  --------------------------
  count(*) as n_edits,
  sum(CASE
  	WHEN has(hashtags, 'mapwithai') THEN 1
  	WHEN edit = 1 THEN 1
  	WHEN editor ILIKE '%rapid%' THEN 1
 	ELSE 0
  END) n_edits_AI,
  sum(CASE
    WHEN arrayExists(
        x -> multiSearchAnyCaseInsensitive(x, humanitarian_hashtags),
        hashtags
    ) THEN 1
    ELSE 0
  END) as n_edits_humanitarian,
  sum(CASE
    WHEN arrayExists(
        x -> multiSearchAnyCaseInsensitive(x, corporate_hashtags),
        hashtags
    ) THEN 1
    ELSE 0
  END) as n_edits_corporate
FROM int.all_stats_user_3
WHERE 1=1
--  and user_id in  [115612, 996790, 23367952, 408282, 9514903]
GROUP BY user_id, month, country
ORDER BY user_id, month, country;
# SemBench query selectivity (highest scale factor per scenario)

selectivity = rows selected by the query / rows in the base table. For aggregations the headline value is the predicate selectivity (fraction of base rows feeding the aggregate); for joins it is the result/base ratio (can exceed 1); for ranking/classification it is ~1.

| scenario | query | type | sf | input rows | result rows | selected rows | selectivity | status | notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| movie | Q1 | filter | 2000 | 2000 | 1487 | 1487 | 74.350% | ok | computed at available sf=2000 (target 16000 not on disk); clearly positive reviews (NL: LIMIT 5) |
| movie | Q2 | filter | 2000 | 2000 | 14 | 14 | 0.700% | ok | computed at available sf=2000 (target 16000 not on disk); positive reviews for taken_3 (NL: LIMIT 5) |
| movie | Q3 | aggregation | 2000 | 2000 | 1 | 14 | 0.700% | ok | computed at available sf=2000 (target 16000 not on disk); COUNT positive reviews for taken_3 |
| movie | Q4 | aggregation | 2000 | 2000 | 1 | 120 | 6.000% | ok | computed at available sf=2000 (target 16000 not on disk); positivity ratio over taken_3 reviews |
| movie | Q5 | join | 2000 | 2000 | 32288 | 256 | 1614.400% | ok | computed at available sf=2000 (target 16000 not on disk); self-join: same-sentiment pairs (ant_man); result=pairs |
| movie | Q6 | join | 2000 | 2000 | 32736 | 256 | 1636.800% | ok | computed at available sf=2000 (target 16000 not on disk); self-join: opposite-sentiment pairs (ant_man), NL: LIMIT 10 |
| movie | Q7 | join | 2000 | 2000 | 32736 | 256 | 1636.800% | ok | computed at available sf=2000 (target 16000 not on disk); self-join: all opposite-sentiment pairs (ant_man) |
| movie | Q8 | aggregation | 2000 | 2000 | 2 | 120 | 6.000% | ok | computed at available sf=2000 (target 16000 not on disk); GROUP BY sentiment for taken_3 |
| movie | Q9 | rank | 2000 | 2000 | 256 | 256 | 12.800% | ok | computed at available sf=2000 (target 16000 not on disk); score each ant_man review (subset of Reviews) |
| movie | Q10 | rank | 2000 | 116 | 116 | 116 | 100.000% | ok | computed at available sf=2000 (target 16000 not on disk); rank every movie (returns all rows) |
| ecomm | Q1 | filter | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q2 | filter | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q3 | map | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q4 | map | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q5 | classification | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q6 | classification | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q7 | join | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q8 | join | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q9 | join | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q10 | join | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q11 | join | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q12 | classification | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q13 | filter | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| ecomm | Q14 | rank | 4000 | - | - | - | - | data unavailable | python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor 4000 |
| cars | Q1 | filter | 157376 | 157376 | 7664 | 7664 | 4.870% | ok | from committed ground-truth (Q1_157376.csv); cars in a crash/accident (text) |
| cars | Q2 | filter | 157376 | 157376 | 9 | 9 | 0.006% | ok | from committed ground-truth (Q2_157376.csv); electric cars with dead-battery audio |
| cars | Q3 | filter | 157376 | 157376 | 5418 | 5418 | 3.443% | ok | from committed ground-truth (Q3_157376.csv); manual cars, no damage in images (NL: LIMIT 10) |
| cars | Q4 | aggregation | 157376 | 157376 | 1 | - | - | ok | from committed ground-truth (Q4_157376.csv); AVG age of cars with engine problems (text) | predicate selectivity needs labelled source data |
| cars | Q5 | aggregation | 157376 | 157376 | 1 | - | - | ok | from committed ground-truth (Q5_157376.csv); COUNT automatic cars damaged in audio AND image | predicate selectivity needs labelled source data |
| cars | Q6 | filter | 157376 | 157376 | 14774 | 14774 | 9.388% | ok | from committed ground-truth (Q6_157376.csv); cars damaged in one modality but not another |
| cars | Q7 | filter | 157376 | 157376 | 31818 | 31818 | 20.218% | ok | from committed ground-truth (Q7_157376.csv); dented(img) OR worn brakes(audio) OR electrical(text) |
| cars | Q8 | filter | 157376 | 157376 | 1459 | 1459 | 0.927% | ok | from committed ground-truth (Q8_157376.csv); punctures AND paint scratches in images (NL: LIMIT 100) |
| cars | Q9 | filter | 157376 | 157376 | 2 | 2 | 0.001% | ok | from committed ground-truth (Q9_157376.csv); torn(img) AND bad ignition(audio) |
| cars | Q10 | classification | 157376 | 157376 | 157269 | 157269 | 99.932% | ok | from committed ground-truth (Q10_157376.csv); classify every complaint into a component class |
| animals | Q1 | aggregation | 200 | 200 | 1 | 11 | 5.500% | ok | computed at available sf=200 (target 1600 not on disk); COUNT zebra images |
| animals | Q2 | aggregation | 200 | 66 | 1 | 5 | 7.576% | ok | computed at available sf=200 (target 1600 not on disk); COUNT elephant audio |
| animals | Q3 | rank | 200 | 200 | 2 | 11 | 1.000% | ok | computed at available sf=200 (target 1600 not on disk); city with most zebra images (result=cities) |
| animals | Q4 | rank | 200 | 66 | 2 | 5 | 3.030% | ok | computed at available sf=200 (target 1600 not on disk); city with most elephant audio (result=cities) |
| animals | Q5 | set | 200 | 266 | 3 | 3 | 1.128% | ok | computed at available sf=200 (target 1600 not on disk); cities with elephant image OR audio (result=cities) |
| animals | Q6 | set | 200 | 266 | 3 | 3 | 1.128% | ok | computed at available sf=200 (target 1600 not on disk); cities with monkey image but NO monkey audio (result=cities) |
| animals | Q7 | set | 200 | 200 | 5 | 5 | 2.500% | ok | computed at available sf=200 (target 1600 not on disk); cities where zebra AND impala co-occur in images (result=cities) |
| animals | Q8 | set | 200 | 266 | 3 | 3 | 1.128% | ok | computed at available sf=200 (target 1600 not on disk); cities with elephant AND monkey across modalities (result=cities) |
| animals | Q9 | set | 200 | 266 | 2 | 2 | 0.752% | ok | computed at available sf=200 (target 1600 not on disk); cities with monkey image AND monkey audio (result=cities) |
| animals | Q10 | rank | 200 | 200 | 2 | 11 | 1.000% | ok | computed at available sf=200 (target 1600 not on disk); city+station with most zebra images (result=tuples) |
| mmqa | Q1 | filter | 800 | 800 | 1 | 1 | 0.125% | ok | base table padded to scale_factor rows; Who is the director of the movie that has Ben Piazza in the  |
| mmqa | Q2A | retrieval | 800 | 800 | 5 | 5 | 0.625% | ok | base table padded to scale_factor rows; Identify the images containing logos, if available, for each |
| mmqa | Q2B | retrieval | 800 | 800 | 5 | 5 | 0.625% | ok | base table padded to scale_factor rows; Identify the images containing logos, if available, for each |
| mmqa | Q3A | filter | 800 | 800 | 13 | 13 | 1.625% | ok | base table padded to scale_factor rows; Which movies are comedies? |
| mmqa | Q3B | filter | 800 | 800 | 3 | 3 | 0.375% | ok | base table padded to scale_factor rows; Which movies are sci-fi? |
| mmqa | Q3C | filter | 800 | 800 | 4 | 4 | 0.500% | ok | base table padded to scale_factor rows; Which movies are romances? |
| mmqa | Q3D | filter | 800 | 800 | 2 | 2 | 0.250% | ok | base table padded to scale_factor rows; Which movies are horror? |
| mmqa | Q3E | filter | 800 | 800 | 1 | 1 | 0.125% | ok | base table padded to scale_factor rows; Which movies are heist movies? |
| mmqa | Q3F | filter | 800 | 800 | 3 | 3 | 0.375% | ok | base table padded to scale_factor rows; Which movies are romantic comedies? |
| mmqa | Q3G | filter | 800 | 800 | 1 | 1 | 0.125% | ok | base table padded to scale_factor rows; Which movies are biographical comedies? |
| mmqa | Q4 | classification | 800 | 800 | 39 | 39 | 4.875% | ok | base table padded to scale_factor rows; Categorize the movies in the table by their genre. If a movi |
| mmqa | Q5 | filter | 800 | 800 | 1 | 1 | 0.125% | ok | base table padded to scale_factor rows; Who has played a role in all the following movies: 'Love Is  |
| mmqa | Q6A | filter | 800 | 800 | 1 | 1 | 0.125% | ok | base table padded to scale_factor rows; Which airlines have destinations in Frankfurt? |
| mmqa | Q6B | filter | 800 | 800 | 1 | 1 | 0.125% | ok | base table padded to scale_factor rows; Which airlines have destinations in Germany? |
| mmqa | Q6C | filter | 800 | 800 | 5 | 5 | 0.625% | ok | base table padded to scale_factor rows; Which airlines have destinations in Europe? |
| mmqa | Q7 | retrieval | 800 | 800 | 5 | 5 | 0.625% | ok | base table padded to scale_factor rows; For each airline with destinations in Europe, find its logo  |
| medical | Q1 | filter | 11112 | 11112 | 50 | 50 | 0.450% | ok | patients with text_diagnosis=allergy |
| medical | Q2 | filter | 11112 | 11112 | 22 | 22 | 0.198% | ok | non-current smokers with normal audio |
| medical | Q3 | filter | 11112 | 11112 | 1357 | 1357 | 12.212% | ok | family cancer & abnormal x-ray (NL: LIMIT 5) |
| medical | Q4 | aggregation | 11112 | 11112 | 1 | 50 | 0.450% | ok | AVG age of acne patients |
| medical | Q5 | aggregation | 11112 | 11112 | 1 | 1 | 0.009% | ok | GROUP BY smoking: current smokers, abnormal audio+xray |
| medical | Q6 | filter | 11112 | 11112 | 286 | 286 | 2.574% | ok | sick in >=1 modality but normal in >=1 (NL: youngest) |
| medical | Q7 | aggregation | 11112 | 11112 | 7594 | 7594 | 68.341% | ok | is_sick patients (NL: AVG age); GT = matching rows |
| medical | Q8 | filter | 11112 | 11112 | 378 | 378 | 3.402% | ok | family cancer & malignant skin (NL: LIMIT 100) |
| medical | Q9 | filter | 11112 | 11112 | 469 | 469 | 4.221% | ok | malignant skin & abnormal x-ray |
| medical | Q10 | classification | 11112 | 11112 | 1200 | 1200 | 10.799% | ok | diagnose symptom text for every patient w/ symptoms |

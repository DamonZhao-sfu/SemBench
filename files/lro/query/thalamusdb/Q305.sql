SELECT
    l.left_colname AS left_colname,
    r.right_colname AS right_colname
FROM lro_q305_left l, lro_q305_right r
WHERE NLjoin(
    l.left_descriptor,
    r.right_descriptor,
    'Determine whether the two columns from different tables describe the same real-world feature (same semantic attribute), using the column names and sample values. Each descriptor is formatted as: name=<colname> | samples=<sample1> | <sample2> | ...'
);

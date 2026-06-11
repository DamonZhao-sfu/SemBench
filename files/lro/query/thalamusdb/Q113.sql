SELECT
    l.left_name AS left_name,
    r.right_name AS right_name
FROM lro_q113_left l, lro_q113_right r
WHERE NLjoin(
    l.left_name,
    r.right_name,
    'Determine whether the two senior government official names share the same first name (first given name, ignoring middle names / hyphenated parts).'
);

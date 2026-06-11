SELECT
    l.left_Supplier AS left_Supplier,
    r.right_Supplier AS right_Supplier
FROM lro_q121_left l, lro_q121_right r
WHERE NLjoin(
    l.left_Supplier,
    r.right_Supplier,
    'Determine whether these two supplier names refer to the same organisation (ignoring case, abbreviations, punctuation, trailing legal suffixes such as Ltd/Limited, and minor formatting differences).'
);

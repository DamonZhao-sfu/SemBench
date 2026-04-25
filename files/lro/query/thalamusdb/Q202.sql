SELECT
    l.left_ID AS left_ID,
    r.right_ID AS right_ID
FROM lro_q202_left l, lro_q202_right r
WHERE NLjoin(
    l.left_description,
    r.right_description,
    'Determine whether the two restaurant records refer to the same real-world restaurant (same establishment at the same address). Each record is formatted as: name=... | address=... | phone=... | cuisine=...'
);

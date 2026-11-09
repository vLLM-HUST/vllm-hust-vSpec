"""Exercise the compiled length-admission function before any NPU context."""


def check(library):
    count = 0
    for n in (1, 2, 3, 14):
        for width in (5, 7):
            rows = n * width
            for lengths in ([width * (i + 1) for i in range(n)], list(range(1, rows + 1))):
                assert library.valid_lengths(rows, lengths, [512] * len(lengths))
                assert not library.valid_lengths(rows, lengths, [512] * (len(lengths) + 1))
                changed = lengths.copy(); changed[-1] -= 1
                assert not library.valid_lengths(rows, changed, [512] * len(changed))
                count += 3
    for rows, lengths in ((28, [7,14,21,28]), (9, [9]), (0, []), (7, [0,7]),
                          (70, [5] * 14), (98, [7] * 14), (14, [14])):
        assert not library.valid_lengths(rows, lengths, [512] * len(lengths))
        count += 1
    assert count == 55
    return count

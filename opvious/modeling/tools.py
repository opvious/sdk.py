def project(t, mask):
    return tuple(t[i] for i, t in enumerate(t) if mask[i])

import numpy as np
import pandas as pd

def parse_calbody(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Nd, Na, Nc = map(int, parts[:3])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    expected = Nd + Na + Nc
    if data.shape[0] != expected:
        raise ValueError(f"CALBODY row mismatch in {path}: got {data.shape[0]}, expected {expected}")

    d = data[0:Nd]
    a = data[Nd:Nd+Na]
    c = data[Nd+Na:Nd+Na+Nc]
    return d, a, c


def parse_calreadings(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Nd, Na, Nc, Nframes = map(int, parts[:4])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    block_rows = Nd + Na + Nc
    expected = block_rows * Nframes
    if data.shape[0] != expected:
        raise ValueError(f"CALREADINGS row mismatch in {path}: got {data.shape[0]}, expected {expected}")

    D_frames, A_frames, C_frames = [], [], []
    for k in range(Nframes):
        start = k * block_rows
        D_frames.append(data[start:start+Nd])
        A_frames.append(data[start+Nd:start+Nd+Na])
        C_frames.append(data[start+Nd+Na:start+Nd+Na+Nc])
    return np.array(D_frames), np.array(A_frames), np.array(C_frames)


def parse_empivot(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Ng, Nframes = map(int, parts[:2])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    expected = Ng * Nframes
    if data.shape[0] != expected:
        raise ValueError(f"EMPIVOT row mismatch in {path}: got {data.shape[0]}, expected {expected}")

    frames = [data[i*Ng:(i+1)*Ng] for i in range(Nframes)]
    return np.array(frames)


def parse_optpivot(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Nd, Nh, Nframes = map(int, parts[:3])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    expected = (Nd + Nh) * Nframes
    if data.shape[0] != expected:
        raise ValueError(f"OPTPIVOT row mismatch in {path}: got {data.shape[0]}, expected {expected}")

    D_frames, H_frames = [], []
    for k in range(Nframes):
        start = k * (Nd + Nh)
        D_frames.append(data[start:start+Nd])
        H_frames.append(data[start+Nd:start+Nd+Nh])
    return np.array(D_frames), np.array(H_frames)


def parse_ct_fiducials(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Nb = int(parts[0])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    if data.shape[0] != Nb:
        raise ValueError(f"CT-FIDUCIALS row mismatch in {path}: got {data.shape[0]}, expected {Nb}")
    return data


def parse_em_fiducials(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Ng, Nb = map(int, parts[:2])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    expected = Ng * Nb
    if data.shape[0] != expected:
        raise ValueError(f"EM-FIDUCIALS row mismatch in {path}: got {data.shape[0]}, expected {expected}")

    frames = [data[i*Ng:(i+1)*Ng] for i in range(Nb)]
    return np.array(frames)


def parse_em_nav(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
    parts = header.replace(',', ' ').split()
    Ng, Nframes = map(int, parts[:2])

    df = pd.read_csv(path, header=None, skiprows=1)
    data = df.iloc[:, :3].to_numpy(float)

    expected = Ng * Nframes
    if data.shape[0] != expected:
        raise ValueError(f"EM-NAV row mismatch in {path}: got {data.shape[0]}, expected {expected}")

    frames = [data[i*Ng:(i+1)*Ng] for i in range(Nframes)]
    return np.array(frames)

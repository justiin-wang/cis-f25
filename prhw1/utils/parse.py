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

if __name__ == "__main__":
    # Verify parsing functions are good by checking shapes
    calbody_path = "prhw1/data/pa1-debug-a-calbody.txt"
    d, a, c = parse_calbody(calbody_path)
    print("D points shape:", d.shape)
    print("A points shape:", a.shape)
    print("C points shape:", c.shape)

    calreadings_path = "prhw1/data/pa1-debug-a-calreadings.txt"
    D_frames, A_frames, C_frames = parse_calreadings(calreadings_path)
    print("D_frames shape:", D_frames.shape)
    print("A_frames shape:", A_frames.shape)
    print("C_frames shape:", C_frames.shape)

    empivot_path = "prhw1/data/pa1-debug-a-empivot.txt"
    empivot_frames = parse_empivot(empivot_path)
    print("Empivot frames shape:", empivot_frames.shape)

    optpivot_path = "prhw1/data/pa1-debug-a-optpivot.txt"
    D_opt_frames, H_frames = parse_optpivot(optpivot_path)
    print("D_opt_frames shape:", D_opt_frames.shape)
    print("H_frames shape:", H_frames.shape)


def parse_output_1(path):
    with open(path, 'r') as f:
        header = f.readline().strip()
        parts = header.replace(',', ' ').split()
        Nc, Nframes = map(int, parts[:2])

        p_post_em = np.fromstring(f.readline(), sep=',')
        p_post_opt = np.fromstring(f.readline(), sep=',')

        data = np.loadtxt(path, delimiter=',', skiprows=3)
        C_expected_frames = data.reshape(Nframes, Nc, 3)

    return C_expected_frames, p_post_em, p_post_opt


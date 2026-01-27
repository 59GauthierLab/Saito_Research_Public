import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import soundfile as sf

# 入出力ディレクトリ
INPUT_DIR = "./input"
TEACHER_DIR = "./teacher"
INPUT_CROPPED_DIR = "../dataset/input_cropped"
TEACHER_CROPPED_DIR = "../dataset/teacher_cropped"
os.makedirs(INPUT_CROPPED_DIR, exist_ok=True)
os.makedirs(TEACHER_CROPPED_DIR, exist_ok=True)

# constants
SR_RATE = 44100  # サンプリングレート


def error_msg_exit(message: str) -> None:
    """
    エラーメッセージを表示して終了
    """
    print(f"[ERROR] {message}")
    exit(1)


def info_msg(message: str) -> None:
    """
    デバッグ用・処理確認用にメッセージを表示
    """
    print(f"[INFO] {message}")


def load_audio(path: str) -> np.ndarray:
    """
    音声ファイルを読み込み，モノラル波形を返す
    """
    try:
        wav_arr, sample_rate = librosa.load(path, sr=None, mono=True)
        info_msg(f"Loaded {path} (sample_rate={sample_rate}, wav_size={len(wav_arr)})")
        if sample_rate != SR_RATE:
            error_msg_exit(
                f"Sampling rate is not {SR_RATE/1000}kHz: {path} (sr={sample_rate})"
            )
        return wav_arr
    except Exception as e:
        error_msg_exit(f"Load audio failed: {path} : {e}")


def find_best_alignment(tgt: np.ndarray, ref: np.ndarray) -> int:
    """
    相互相関から正しいずらし位置(lag)を見つける
    tgt が k で動き，ref に合わせる
    tgt[-|ref|+1] = ... = tgt[-1] = 0 と仮定 (式的に負の配列外参照は0扱いとする)
    ref はそのまま
    """
    tgt = tgt - np.mean(tgt)
    ref = ref - np.mean(ref)

    # 相互相関計算
    # corr[k] = \sum_{i=0}^{|ref| - 1} (k=0,...,|tgt| + |ref| - 2)
    #            tgt[k - (|ref| - 1) + i] * ref[i]
    #         = tgt[k + 1 - |ref|] * ref[0] +
    #           tgt[k + 2 - |ref|] * ref[1] + ... +
    #           tgt[k] * ref[|ref| - 1]
    # if k = 0
    # corr[0] = tgt[- |ref| + 1] * ref[0] +
    #           tgt[- |ref| + 2] * ref[1] + ... +
    #           tgt[0] * ref[|ref| - 1]
    # if k = |tgt| + |ref| - 2
    # corr[|tgt| + |ref| - 2] = tgt[|tgt| - 1] * ref[0] +
    #                           tgt[|tgt|] * ref[1] + ... +
    #                           tgt[|tgt| + |ref| - 2] * ref[|ref| - 1]
    corr = np.correlate(tgt, ref, mode="full")

    # 相互相関が最大となるようなずらしを取得
    imax = np.argmax(corr)

    # (lag >= 0) tgt[lag:] と ref[0:] が一致
    # (lag <  0) tgt[0:] と ref[-lag:] が一致 (Note: -lag > 0)
    lag = imax - (len(ref) - 1)

    return lag


def crop_common_region(
    tgt: np.ndarray, ref: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    ずらし位置(lag)に基づいて共通領域をクロップする

    Args:
        tgt (np.ndarray): 対象信号
        ref (np.ndarray): 参照信号
        lag (int):
            lag >= 0 => tgt[lag:] と ref[0:]    が一致
            lag <  0 => tgt[0:]   と ref[-lag:] が一致 (Note: -lag > 0)
    """
    tgt_length, ref_length = len(tgt), len(ref)

    # tgt_start, ref_start >= 0
    if lag >= 0:
        tgt_start = lag
        ref_start = 0
    else:
        tgt_start = 0
        ref_start = -lag

    # L: 共通領域の長さ
    L = min(tgt_length - tgt_start, ref_length - ref_start)

    # 共通領域をクロップ (長さL)
    tgt_cropped = tgt[tgt_start : tgt_start + L]
    ref_cropped = ref[ref_start : ref_start + L]

    return tgt_cropped, ref_cropped


def process_pair(input_file_path: str, teacher_file_path: str, file_name: str) -> None:
    """
    入力音声と教師音声のずれをなくし，その共通部分を保存する

    Args:
        input_file_path: 入力音声ファイルのパス
        teacher_file_path: 教師音声ファイルのパス
        file_name: ファイル名 (共通)
    """
    info_msg(f"=== Processing {file_name} ===")
    input_wav_arr = load_audio(input_file_path)
    teacher_wav_arr = load_audio(teacher_file_path)

    # 音声が重なるずらし(lag)を計算
    info_msg(f"Finding alignment for {file_name}...")
    lag = find_best_alignment(input_wav_arr, teacher_wav_arr)
    info_msg(f"lag={lag} samples ({lag/SR_RATE:.3f}s)")

    # 共通領域をクロップ
    info_msg("Cropping common region...")
    input_cropped, teacher_cropped = crop_common_region(
        input_wav_arr, teacher_wav_arr, lag
    )
    info_msg(f"Cropped sizes: {len(teacher_cropped)} samples")  # same size for both

    input_cropped_path = os.path.join(INPUT_CROPPED_DIR, file_name)
    teacher_cropped_path = os.path.join(TEACHER_CROPPED_DIR, file_name)

    info_msg(f"Writing cropped files for {file_name}...")
    sf.write(teacher_cropped_path, teacher_cropped, SR_RATE)
    sf.write(input_cropped_path, input_cropped, SR_RATE)
    info_msg("=== Saved successfully! ===")

    return


def main():
    teacher_file_names = sorted(
        [f for f in os.listdir(TEACHER_DIR) if f.endswith(".wav")]
    )
    if not teacher_file_names:
        error_msg_exit("No .wav files found in ./teacher dir")

    input_file_names = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")])
    if not input_file_names:
        error_msg_exit("No .wav files found in ./input dir")

    if teacher_file_names != input_file_names:
        error_msg_exit(".wav file names in ./teacher and ./input do not match")

    info_msg(f"=== start processing {len(teacher_file_names)} files in parallel ===")

    # 並列処理で各 wav ペアを処理
    with ProcessPoolExecutor() as executor:
        futures = {}
        for file_name in teacher_file_names:
            input_file_path = os.path.join(INPUT_DIR, file_name)
            teacher_file_path = os.path.join(TEACHER_DIR, file_name)
            # LHS: process_pair を並列実行する Future オブジェクト
            # RHS: file_name (何を実行したかの識別用．下記エラーハンドリング時に使用)
            futures[
                executor.submit(
                    process_pair, input_file_path, teacher_file_path, file_name
                )
            ] = file_name

    for future in as_completed(futures):
        file_name = futures[future]
        try:
            future.result()
        except Exception as e:
            error_msg_exit(f"Processing failed for {file_name}: {e}")

    info_msg("=== all processing done! ===")


if __name__ == "__main__":
    main()

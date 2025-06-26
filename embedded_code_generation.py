#   Copyright 2025 BirdNET-Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import keras
import tensorflow as tf

from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESP_IDF_v5_2
from biodcase_tiny.feature_extraction.feature_extraction import make_constants
from config import Config, load_config
from paths import KERAS_MODEL_PATH, REFERENCE_DATASET_PATH, GEN_CODE_DIR, TFLITE_MODEL_PATH


def create_target(model, reference_dataset: tf.data.Dataset, config: Config):
    dp_c = config.data_preprocessing
    fe_c = config.feature_extraction
    feature_config = make_constants(
        sample_rate=dp_c.sample_rate,
        win_samples=fe_c.window_len, window_scaling_bits=fe_c.window_scaling_bits,
        mel_n_channels=fe_c.mel_n_channels, mel_low_hz=fe_c.mel_low_hz, mel_high_hz=fe_c.mel_high_hz,
        mel_post_scaling_bits=fe_c.mel_post_scaling_bits
    )
    target = ESPTarget(model, feature_config, reference_dataset)
    target.validate()
    return target


def run_embedded_code_generation(config: Config):
    model = keras.models.load_model(KERAS_MODEL_PATH)
    reference_ds = tf.data.Dataset.load(str(REFERENCE_DATASET_PATH))
    target = create_target(model, reference_ds, config)

    tflite_model_buf = target.get_model_buf(model, reference_ds)
    with TFLITE_MODEL_PATH.open("wb") as f:
        f.write(tflite_model_buf)

    toolchain = ESP_IDF_v5_2(config.embedded_code_generation.serial_device)
    src_path = GEN_CODE_DIR / "src"
    src_path.mkdir(exist_ok=True)

    target.process_target_templates(src_path)
    toolchain.compile(src_path=src_path)
    toolchain.flash(src_path=src_path)
    toolchain.monitor(src_path=src_path)


if __name__ == '__main__':
    config = load_config()
    run_embedded_code_generation(config)

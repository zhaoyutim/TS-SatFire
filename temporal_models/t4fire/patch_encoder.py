# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from scipy import spatial

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            "num_patches":self.num_patches,
            "position_embedding": self.position_embedding,
        })

        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.position_embedding(positions)


if __name__=='__main__':
    patch_encoder = PatchEncoder(10, 192)
    patch = np.zeros((10,45))
    patch_encoder.call(patch)

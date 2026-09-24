"""Verify the saved model can reproduce scientific predictions, not merely upload."""
import numpy as np
import pytest
tf = pytest.importorskip('tensorflow')
qkeras = pytest.importorskip('qkeras')
from tests.helpers import make_tracker
import rheed_runtime


def test_quantized_model_snapshot_reproduces_predictions(tmp_path):
    x = tf.keras.Input(shape=(2,))
    y = qkeras.QDense(2, kernel_quantizer=qkeras.quantized_bits(8,0,alpha=1))(x)
    y = qkeras.QActivation(qkeras.quantized_relu(8,0))(y)
    model = tf.keras.Model(x,y)
    model.set_weights([np.array([[.25,.5],[.5,-.25]],dtype=np.float32),np.zeros(2,dtype=np.float32)])
    samples = np.array([[.5,.25],[.25,.5]],dtype=np.float32)
    expected = model(samples,training=False).numpy()
    tracker, client, _ = make_tracker(tmp_path)
    tracker.record_cell(source='model = train()',assigned_names=['model'],user_ns={'model':model})
    path=next(r['path'] for r in client.uploads if r['path'].suffix=='.npz')
    restored=rheed_runtime.restore_model_snapshot(path)
    np.testing.assert_array_equal(restored(samples,training=False).numpy(),expected)

"""Recompute full-dimensional drift and the fixed PCA from saved kernels."""
import matplotlib.pyplot as plt
import numpy as np
from . import paths
from .common import read_data, save_json
from .direction_panels import save_split_panels


def build():
    reference=read_data('trajectory_reference.json')
    epochs=np.array(reference['epochs'])
    seeds=reference['seeds']
    with np.load(paths.ROOT/'data/kernels.npz', allow_pickle=False) as data:
        weights=np.array([[data[f's{s}_e{e:03d}'] for e in epochs] for s in seeds],dtype=float)
    assert weights.shape==(4,26,300,600)
    norms=np.linalg.norm(weights,axis=-1,keepdims=True)
    assert np.all(norms>0) and np.isfinite(weights).all()
    unit=weights/norms
    angles=np.degrees(np.arccos(np.clip((unit*unit[:,:1]).sum(-1),-1,1))).mean(-1)
    angles[:,0]=0
    consecutive=np.degrees(np.arccos(np.clip((unit[:,1:]*unit[:,:-1]).sum(-1),-1,1))).mean(-1)
    np.testing.assert_allclose(angles,reference['reference_angles'],atol=1e-9)
    np.testing.assert_allclose(consecutive[:,:-1],reference['four_epoch_angles'],atol=1e-9)
    example=unit[seeds.index(reference['example_seed'])]
    flat=example.reshape(-1,600)
    center=flat.mean(0)
    # Fit one shared basis; resolve arbitrary SVD signs against published axes.
    _, singular, components=np.linalg.svd(flat-center,full_matrices=False)
    components=components[:3]
    signs=np.sign(np.sum(components*np.array(reference['pca_components']),axis=1))
    components*=signs[:,None]
    np.testing.assert_allclose(center,reference['pca_mean'],atol=1e-10)
    np.testing.assert_allclose(components,reference['pca_components'],atol=1e-8)
    explained=singular[:3]**2/(singular**2).sum()
    np.testing.assert_allclose(explained,reference['pca_explained_variance'],atol=1e-10)
    coords=(example-center)@components.T
    selected=np.random.default_rng(reference['sampling_seed']).choice(300,8,replace=False)[:5].tolist()
    assert selected==reference['selected_neurons']
    shown=reference['snapshot_epochs']
    indices=[int(np.where(epochs==e)[0][0]) for e in shown]
    assert np.max(np.abs(coords[indices,:,:2]))<reference['axis_limit']
    computed={'reference_angles':angles.tolist(),'four_epoch_angles':consecutive[:,:-1].tolist()}
    colors=plt.get_cmap('tab20')(list(range(0,20,2))+list(range(1,20,2)))
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    save_split_panels(paths.OUT,coords,selected,shown,indices,reference['axis_limit'],colors,computed,epochs)
    save_json('trajectory_statistics.json',dict(**computed,selected_neurons=selected,
        pca_explained_variance=explained.tolist(),pca_refitted=True,example_seed=reference['example_seed']))
    print('PASS: full-dimensional angles, PCA refit and fixed random sample match')

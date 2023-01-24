bvecs = np.array(
    [
        [0.283, 0.283, 0.917],
        [-0.283, 0.283, 0.917],
        [-0.283, -0.283, 0.917],
        [0.283, -0.283, 0.917],
        [0.689, 0.285, 0.667],
        [0.285, 0.689, 0.667],
        [-0.285, 0.689, 0.667],
        [-0.689, 0.285, 0.667],
        [-0.689, -0.285, 0.667],
        [-0.285, -0.689, 0.667],
        [0.285, -0.689, 0.667],
        [0.689, -0.285, 0.667],
        [0.943, 0.0, 0.333],
        [0.667, 0.667, 0.333],
        [0.0, 0.943, 0.333],
        [-0.667, 0.667, 0.333],
        [-0.943, 0.0, 0.333],
        [-0.667, -0.667, 0.333],
        [-0.0, -0.943, 0.333],
        [0.667, -0.667, 0.333],
        [0.924, 0.383, 0.0],
        [0.383, 0.924, 0.0],
        [-0.383, 0.924, 0.0],
        [-0.924, 0.383, 0.0],
    ]
)

bvecs = np.concatenate((bvecs, -bvecs))

bvals = np.concatenate((np.ones(len(bvecs)), 2 * np.ones(len(bvecs))))

bvecs = np.vstack((bvecs, bvecs))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(bvals * bvecs[:, 0], bvals * bvecs[:, 1], bvals * bvecs[:, 2])
ax.set_axis_off()
fig.tight_layout()
plt.show()

gradient = mmsim.Gradient(bvals, bvecs)

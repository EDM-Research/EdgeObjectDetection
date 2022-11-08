import numpy as np
from imgaug.augmenters import meta
from imgaug import parameters
from utils import edge_operations


class EdgeAugment(meta.Augmenter):
    def __init__(self, sigma=(0.0, 1.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(EdgeAugment, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.sigma = parameters.handle_continuous_param(
            sigma, "sigma", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True)

        # epsilon value to estimate whether sigma is sufficently above 0 to
        # apply the blur
        self.eps = 1e-3

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,),
                                          random_state=random_state)
        for i, (image, sig) in enumerate(zip(images, samples)):
            image = edge_operations.random_canny(image)
            image = edge_operations.deform_edges(image, deform_factor=sig)
            batch.images[i] = np.expand_dims(image, axis=-1)
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.sigma]

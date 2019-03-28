import pytest

from vegans import MMGAN, WGAN, WGANGP
from .conftest import nz


class TestIntegration(object):
    @pytest.mark.parametrize('nn', [
        MMGAN,
        WGAN,
        WGANGP,
    ])
    def test_run(self, device, nn, generator, critic, gaussian_dataloader):
        nn(generator, critic, gaussian_dataloader, nz=nz, device=device).train()

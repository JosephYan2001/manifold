"""Preserved mathematical checks for the active navigation updater."""
import unittest
import torch
from manifold_project.experiments.cooperative_navigation.training.ppo import returns, gae


class NavigationMathTests(unittest.TestCase):
    def test_terminal_returns_and_gae(self):
        r = torch.tensor([[1.,2.,3.]])
        values = torch.tensor([[.4,.5,.6]])
        expected = torch.tensor([[1+2*.9+3*.81,2+3*.9,3.]])
        torch.testing.assert_close(returns(r,.9),expected)
        torch.testing.assert_close(gae(r,values,.9,1),expected-values)
        self.assertAlmostEqual(float(gae(r,values,.9,.95)[0,-1]),2.4,places=5)

    def test_fisher_equals_expected_sampled_and_gradient(self):
        mu = torch.tensor([.1,.2,.3,.15,.25],dtype=torch.float64)
        q = torch.tensor([.3,-.5,.1,.4,-.3],dtype=torch.float64,requires_grad=True)
        fisher = (mu*q.square()).sum()-(mu*q).sum().square()
        sampled = (mu*(q-(mu*q).sum()).square()).sum()
        torch.testing.assert_close(fisher,sampled)
        torch.testing.assert_close(torch.autograd.grad(fisher,q,retain_graph=True)[0],torch.autograd.grad(sampled,q)[0])

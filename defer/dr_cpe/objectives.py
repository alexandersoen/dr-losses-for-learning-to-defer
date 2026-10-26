from defer.dr_cpe.config import Criterion, DRCPEConfig, to_torch_criterion
from loss.partial import KLIEPLoss, LogisticLoss, LSIFLoss, SquareLoss

DR_CPE_KLIEP = DRCPEConfig(
  criterion=Criterion.CROSS_ENTROPY,
  partial_loss_factory=KLIEPLoss,
)
DR_CPE_LSIF = DRCPEConfig(
  criterion=Criterion.CROSS_ENTROPY,
  partial_loss_factory=LSIFLoss,
)
DR_CPE_LOGISTIC = DRCPEConfig(
  criterion=Criterion.CROSS_ENTROPY,
  partial_loss_factory=LogisticLoss,
)
DR_CPE_SQUARE = DRCPEConfig(
  criterion=Criterion.CROSS_ENTROPY,
  partial_loss_factory=SquareLoss,
)
DR_CPE_MAX_LOG_PROB_KLIEP = DRCPEConfig(
  criterion=Criterion.MAX_PROB_CROSS_ENTROPY,
  partial_loss_factory=KLIEPLoss,
)
DR_CPE_MAX_LOG_PROB_LSIF = DRCPEConfig(
  criterion=Criterion.MAX_PROB_CROSS_ENTROPY,
  partial_loss_factory=LSIFLoss,
)
DR_CPE_MAX_LOG_PROB_LOGISTIC = DRCPEConfig(
  criterion=Criterion.MAX_PROB_CROSS_ENTROPY,
  partial_loss_factory=LogisticLoss,
)
DR_CPE_MAX_LOG_PROB_SQUARE = DRCPEConfig(
  criterion=Criterion.MAX_PROB_CROSS_ENTROPY,
  partial_loss_factory=SquareLoss,
)
DR_CPE_MAX_PROB_KLIEP = DRCPEConfig(
  criterion=Criterion.MAX_NEG_PROB,
  partial_loss_factory=KLIEPLoss,
)
DR_CPE_MAX_PROB_LSIF = DRCPEConfig(
  criterion=Criterion.MAX_NEG_PROB,
  partial_loss_factory=LSIFLoss,
)
DR_CPE_MAX_PROB_LOGISTIC = DRCPEConfig(
  criterion=Criterion.MAX_NEG_PROB,
  partial_loss_factory=LogisticLoss,
)
DR_CPE_MAX_PROB_SQUARE = DRCPEConfig(
  criterion=Criterion.MAX_NEG_PROB,
  partial_loss_factory=SquareLoss,
)
DR_CPE_MATSUSHITA_KLIEP = DRCPEConfig(
  criterion=Criterion.MATSUSHITA,
  partial_loss_factory=KLIEPLoss,
)
DR_CPE_MATSUSHITA_LSIF = DRCPEConfig(
  criterion=Criterion.MATSUSHITA,
  partial_loss_factory=LSIFLoss,
)
DR_CPE_MATSUSHITA_LOGISTIC = DRCPEConfig(
  criterion=Criterion.MATSUSHITA,
  partial_loss_factory=LogisticLoss,
)

__all__ = [
  "Criterion",
  "DRCPEConfig",
  "DR_CPE_KLIEP",
  "DR_CPE_LOGISTIC",
  "DR_CPE_LSIF",
  "DR_CPE_MATSUSHITA_KLIEP",
  "DR_CPE_MATSUSHITA_LOGISTIC",
  "DR_CPE_MATSUSHITA_LSIF",
  "DR_CPE_MAX_LOG_PROB_KLIEP",
  "DR_CPE_MAX_LOG_PROB_LOGISTIC",
  "DR_CPE_MAX_LOG_PROB_LSIF",
  "DR_CPE_MAX_LOG_PROB_SQUARE",
  "DR_CPE_MAX_PROB_KLIEP",
  "DR_CPE_MAX_PROB_LOGISTIC",
  "DR_CPE_MAX_PROB_LSIF",
  "DR_CPE_MAX_PROB_SQUARE",
  "DR_CPE_SQUARE",
  "to_torch_criterion",
]

Faster R-CNN
============

.. module:: chainercv.links.model.faster_rcnn


FasterRCNN
----------
.. autoclass:: FasterRCNN
   :members:
   :special-members:  __call__

FasterRCNNTrainChain
--------------------
.. autoclass:: FasterRCNNTrainChain

FasterRCNNVGG16
---------------
.. autoclass:: FasterRCNNVGG16

VGG16FeatureExtractor
---------------------
.. autoclass:: VGG16FeatureExtractor

VGG16RoIHead
------------
.. autoclass:: VGG16RoIHead


RegionProposalNetwork
---------------------
.. autoclass:: RegionProposalNetwork
   :members:
   :special-members:  __call__


AnchorTargetCreator
"""""""""""""""""""
.. autoclass:: AnchorTargetCreator
   :members:
   :special-members:  __call__

bbox2loc
--------
.. autofunction:: bbox2loc

generate_anchor_base
--------------------
.. autofunction:: generate_anchor_base

loc2bbox
--------
.. autofunction:: loc2bbox

ProposalCreator
---------------
.. autoclass:: ProposalCreator
   :members:
   :special-members:  __call__

ProposalTargetCreator
"""""""""""""""""""""
.. autoclass:: ProposalTargetCreator
   :members:
   :special-members:  __call__

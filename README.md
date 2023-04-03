### Paper_TNSM
**Code for paper Contrastive Learning Enhanced Intrusion Detection



### Abstract
Contrastive Learning Enhanced Intrusion Detection
With the continuous development of network technology, the diversity of network traffic constantly increased (intra-class diversity). Nevertheless, the boundary between malicious and benign actions became even ambiguous (inter-class similarity), causing lots of false detection and hindering the further optimization of the detection model. Focusing on challenges brought by intra-class diversity and inter-class similarity, we proposed a novel approach to enhance intrusion detection based on contrastive learning, which can make the right decision while disentangling samples from different classes. First, to bridge the gap when applying contrastive learning to intrusion detection data, we proposed a heuristic method to build contrastive tasks based on random masking of network packet sequences, which can reflect semantic relationships among samples. Then contrastive loss can be calculated to measure the inter-class and intra-class distances. Second, contrastive cross-entropy loss was proposed, which was a combination of contrastive loss and classification loss. Together with a dual branch deep structure, we can optimize the detection and sample distance requirements at the same time. Thirdly, experiments were conducted on diverse real-world and benchmark datasets using different model architectures under various parameter settings. Results on real-world dataset showed that our methods could stable experience a 5% increase in accuracy, and an 8% improvement in detection rate on those easily misdetected scenarios. To verify the method’s effectiveness on different traffic representations, we further conducted experiments on NSL-KDD and UNSW-NB15, which achieved a 7% accuracy improvement on NSL-KDD and a 6% accuracy improvement on UNSW-NB15. Extensive comparison with state of art intrusion detection models in recent five years showed that the proposed methods could effectively improve the accuracy of detection models.


Acknowledgments
===
It is an implemenation of the algorithm described in *Yue Y, Chen X, Han Z, et al. Contrastive Learning Enhanced Intrusion Detection[J]. IEEE Transactions on Network and Service Management, 2022.*

Please cite our paper when you used our code

To get better result, you need to fine-tune the learning rate, the temperature, the masked_num

More advanced data augmentation methods will bring better performance such methods used in UNSW-NB15


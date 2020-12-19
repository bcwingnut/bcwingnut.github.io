---
title: A Survey on Driving Scenario Generation
published: True
---

## Abstract

Previous works on driving scenario generation include one or several of the following aspects: (i) a scenario library (dataset) containing realistic data, (ii) a method to generate text-based traffic data, which may or may not use existing scenarios, (iii) a method to render images given text-based traffic data.

## Datasets

J. Guo et al. [[1]](https://www.zotero.org/google-docs/?XJrbKp) concluded 45 publicly available datasets for autonomous driving and categorize them according to the tasks that they are suitable for. The datasets are shown in table 1.

TrafficNet [[2]](https://www.zotero.org/google-docs/?XskVZB) is an open naturalistic driving scenario library which applies a set of categorization algorithms to label the naturalistic driving data with six different critical driving scenarios: free flow, pedestrian crossing, cyclist, car-following, lane change, and cut in.

Table 1: Overview of publicly open datasets for autonomous driving.

<img src="/assets/autonomous_driving_datasets.png" width="100%" />

## Scenario Generation

According to W. Wang et al. [[3]](https://www.zotero.org/google-docs/?CJn0n2), however, specific scenarios are not able to cover the entire traffic case, and also not might be suitable to learn for algorithms. More specifically, these manually extracted scenarios are not flexible to be cascaded or combined to generate new traffic scenarios. They proposed a new framework to generate new traffic scenarios from a handful of limited traffic data. This method consists of four steps: primitive extraction, learning primitive sets, topology modeling between primitive sets, and generate new traffic scenarios using primitives.

DeepRoad [[4]](https://www.zotero.org/google-docs/?j2MvC5) is an unsupervised GAN-based framework to automatically generate large amounts of accurate driving scenes to test the consistency of DNN-based autonomous driving systems across different scenes. DeepRoad leverages UNIT [[5]](https://www.zotero.org/google-docs/?zh46WF), a recent published DNN-based method to perform unsupervised image-to-image transformation. DeepRoad firstly takes unpaired training images from two target domains (e.g., one fine driving scene dataset and one rainy driving scene dataset), and utilizes the unsupervised UNIT to map the two scene domains to the same latent space.

According to Menzel et al. [[6]](https://www.zotero.org/google-docs/?ydHZZE), these approaches follow the idea of a data-driven test process within the scope of virtual testing. A generation of synthetic roads or scenarios based on formalized knowledge is not described. Hence, those data-based approaches provide no information regarding the diversity of scenarios. Data-driven approaches can be combined with a knowledge-driven approach. The main idea is to generate scenarios from existing knowledge, like regulatory recommendations, and later augment these synthetic scenarios with information collected from measurement data.

T. Menzel et al. [[7]](https://www.zotero.org/google-docs/?3vGxdf) proposed an approach for a knowledge-based scene creation for automated vehicles based on ontologies. They introduced a nonparametric Bayesian learning method to deal with the challenges in the first two steps, i.e., extracting primitives from multiscale traffic scenarios, where the binary and continuous events are both involved, and obtain the object sets. Driving data used in this paper are extracted from the Safety Pilot Model Deployment (SPMD) database. A Gaussian model is used to generate the observation variables.

## Languages

Scenic [[8]](https://www.zotero.org/google-docs/?XHtgsB) is a probabilistic programming language that allows assigning distributions to features of the scene, as well as declaratively imposing hard and soft constraints over the scene. Scenic can be used to generate specialized test sets, improve the effectiveness of training sets by emphasizing difficult cases, and generalize from individual failure cases to broader scenarios suitable for retraining.

GeoScenario [[9]](https://www.zotero.org/google-docs/?hYCwuD) is a Domain-Specific Language (DSL) for scenario representation to substantiate test cases in simulation. The language was built on top of the well-known Open Street Map standard and designed to be simple and extensible. It is simple and human readable, yet be able to represent precise trajectories collected from traffic data, support input space exploration from methods generating scenarios, and also support unknown stochastic behavior for sampling methods. Exports can manually design scenarios based on functional requirements and designs and hazard analysis, reproduce or augment situations collected from traffic data, or scenario systematically generate scenarios. GeoScenario does not define actions or maneuvers for the Ego but only specifies initial conditions and goals.

## Image Rendering

Manually assigning a class or instance label to every pixel in an image is possible but tedious, requiring up to one hour per image. Thus existing real-world datasets are limited to a few hundred or thousand annotated examples, thereby severely limiting the diversity of the data. In contrast, the creation of virtual 3D environments allows for arbitrary variations of the data and virtually infinite numbers of training samples.

An improviser is implemented for Scenic [[8]](https://www.zotero.org/google-docs/?dEpkWm) scenarios and is used to generate scenes which were rendered into images by Grand Theft Auto V (GTAV [15]), a video game with high-fidelity graphics. The implementation’s interface to GTAV is based on DeepGTAV.

H. A. Alhaija et al. [[10]](https://www.zotero.org/google-docs/?eshaff) proposed a new paradigm for efficiently enlarging existing data distributions using augmented reality. This approach requires three components:  (i) detailed high quality 3D models of cars, (ii) a set of 3D locations and poses used to place the car models in the scene and, (iii) the environment map of the scene that can be used to produce realistic reflections and lighting on the models that matches the scene.

## References

[[1] “Is it Safe to Drive? An Overview of Factors, Challenges, and Datasets for Driveability Assessment in Autonomous Driving.” [Online]. Available: https://arxiv.org/pdf/1811.11277. [Accessed: 31-May-2019].](https://www.zotero.org/google-docs/?3D7Kb9)

[[2] D. Zhao, Y. Guo, and Y. J. Jia, “TrafficNet: An Open Naturalistic Driving Scenario Library,” ArXiv170801872 Cs, Jul. 2017.](https://www.zotero.org/google-docs/?3D7Kb9)

[[3] W. Wang and D. Zhao, “Extracting Traffic Primitives Directly from Naturalistically Logged Data for Self-Driving Applications,” IEEE Robot. Autom. Lett., vol. 3, no. 2, pp. 1223–1229, Apr. 2018.](https://www.zotero.org/google-docs/?3D7Kb9)

[[4] M. Zhang, Y. Zhang, L. Zhang, C. Liu, and S. Khurshid, “DeepRoad: GAN-based Metamorphic Autonomous Driving System Testing,” ArXiv180202295 Cs, Feb. 2018.](https://www.zotero.org/google-docs/?3D7Kb9)

[[5] “Unsupervised Image-to-Image Translation Networks.” [Online]. Available: https://arxiv.org/abs/1703.00848. [Accessed: 31-May-2019].](https://www.zotero.org/google-docs/?3D7Kb9)

[[6] T. Menzel, G. Bagschik, L. Isensee, A. Schomburg, and M. Maurer, “From Functional to Logical Scenarios: Detailing a Keyword-Based Scenario Description for Execution in a Simulation Environment,” ArXiv190503989 Cs, May 2019.](https://www.zotero.org/google-docs/?3D7Kb9)

[[7] T. Menzel, G. Bagschik, and  and M. Maurer, “Scenarios for Development, Test and Validation of Automated Vehicles,” in 2018 IEEE Intelligent Vehicles Symposium (IV), Changshu, 2018, pp. 1821–1827.](https://www.zotero.org/google-docs/?3D7Kb9)

[[8] “Scenic: Language-Based Scene Generation.” [Online]. Available: https://arxiv.org/abs/1809.09310. [Accessed: 26-May-2019].](https://www.zotero.org/google-docs/?3D7Kb9)

[[9] R. Queiroz, T. Berger, and K. Czarnecki, “GeoScenario: An Open DSL for Autonomous Driving Scenario Representation,” p. 8.](https://www.zotero.org/google-docs/?3D7Kb9)

[[10] H. A. Alhaija, S. K. Mustikovela, L. Mescheder, A. Geiger, and C. Rother, “Augmented Reality Meets Computer Vision : Efficient Data Generation for Urban Driving Scenes,” ArXiv170801566 Cs, Aug. 2017.](https://www.zotero.org/google-docs/?3D7Kb9)

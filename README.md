# DTMI

In this paper, we conduct an abstract modeling of existing discrete sensing systems and present the upper and lower bounds of the expected system error based on discrete task mutual information. If it is helpful to you, you are welcome to cite our [paper](https://link.springer.com/article/10.1007/s11432-024-4374-y).

```bibtex
@article{shang2025measuring,
  title={Measuring discrete sensing capability for ISAC via task mutual information},
  author={Shang, Fei and Du, Haohua and Yang, Panlong and He, Xin and Wang, Jingjing and Li, Xiang-Yang},
  journal={Science China Information Sciences},
  volume={68},
  number={5},
  pages={150308},
  year={2025},
  publisher={Science China Press Beijing}
}
```

The calculation of mutual information involved in the paper is implemented using Library [Associations.jl](https://github.com/JuliaDynamics/Associations.jl) in the Julia programming language. This code repository mainly contains the four case studies presented in the supplementary files.

- Human Detection. We deployed several ESP32 nodes in the space, and then used CSI data to build a presence detection instance based on the threshold method. We provided an [example code](./human%20dection/code/plots.py) for data processing and threshold discrimination, as well as a [demo](./human%20dection/demo/human%20dection.mp4).

- Electrical cabinet door state monitoring. This is an implementation example based on RFID. However, due to the involvement of a commercial project, the specific code is not convenient to be made public. Here, we provide a [test demo](./cabinet%20door%20state%20monitoring/door.mov).
- Direction estimation. We generate radio - frequency data using the Method of Moments (MoM) based electromagnetic simulation, then perform angle estimation with the MUSIC algorithm, and finally discriminate the angle intervals. The calculation code for the Method of Moments is in another [independent repository](https://github.com/zaoanhh/EM_simulation_MoM). The MUSIC algorithm is a classic one, and we provide an [example code](./direction%20estimation/music.jl) for it.
- Device type identification. For this part, we referred to an [open-source](https://github.com/YanDawei101/WiFi_device_fingerprint) device identification system. We conducted a screening of their device types, enabling the final detection rate to reach 100% so as to facilitate the verification of the sufficient condition for lossless sensing.
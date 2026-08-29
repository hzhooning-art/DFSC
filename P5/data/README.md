# Public data provenance

The public datasets used in the P5 experiments are not mirrored in this
repository. They are downloaded from their authoritative repositories and
verified before use:

| Key | Dataset | Persistent identifier | Frozen archive SHA-256 |
| --- | --- | --- | --- |
| `brain-corona` | Brain-tissue cyclic relaxation, corona radiata | `10.5281/zenodo.13960486` | `881be97c59d2a212aca19b0c658e692dac2a561296abffaf91b5d934f3607a9a` |
| `brain-visual` | Brain-tissue cyclic relaxation, visual cortex | `10.5281/zenodo.13960486` | `c8e06b910cb3c17d7dde8648d011100027005b5ef9e091a069744d2f8657dc01` |
| `copper` | C19010 copper-alloy stress relaxation | `10.5281/zenodo.10796926` | `44ec89de2d263dde3e67b0309e61cb7a4655a509a06d08954e348b704c66c2ed` |
| `steel-270` | Martensitic-steel relaxation, 270 MPa | `10.5281/zenodo.14051050` | `d567f75f13e9a5652fcbf7819fedbb2c66346d15a724b99cd9ca11634e45aaa3` |
| `steel-300` | Martensitic-steel relaxation, 300 MPa | `10.5281/zenodo.14051050` | `53b1f8b361c155a3964e615150c5901c62981030d412bbe3932b43627b3becc1` |
| `pva` | PVA stress-relaxation curves | `10.5281/zenodo.21333840` | `70cfe35e93ee1421fc5ae4c752f61d11fe83972ae36d2d4321d8e511feeb470f` |
| `kupfer` | KupferDigital stress-relaxation tests | `10.5281/zenodo.10820438` | `4af15b14f0120c58ffee4a2c716a0457450d7819a04720d8723b630b780212f9` |
| `uci-gas` | Gas sensor array under flow modulation | `10.24432/C5BG7G` | `7b062960dbef1a9e8aefc62bc7dbac09bdadb8cbb40793cf2b8e122da1862f90` |
| `uci-hydraulic` | Condition monitoring of hydraulic systems | `10.24432/C5CW21` | `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2` |

Prepare all datasets from the repository root with:

```console
python P5/scripts/fetch_public_data.py --dataset all
```

Use `--dataset KEY` with any key in the table to retrieve one dataset. The
script rejects a download whose SHA-256 digest differs from the frozen source.
Dataset licenses and attribution requirements remain those of the original
repositories.

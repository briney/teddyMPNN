# Bundled pretrained weights — third-party notices

The `.pt` files in this directory are redistributed under the MIT License from
the original ProteinMPNN and LigandMPNN releases by Justas Dauparas and
collaborators (Institute for Protein Design, University of Washington).

## Files

| File | Source repo | Variant |
|---|---|---|
| `proteinmpnn_v_48_020.pt` | https://github.com/dauparas/ProteinMPNN | 48 hidden dim, 0.20 Å backbone noise |
| `ligandmpnn_v_32_010_25.pt` | https://github.com/dauparas/LigandMPNN | 32 hidden dim, 0.10 Å backbone noise, 25-atom ligand context |

Original distribution: https://files.ipd.uw.edu/pub/ligandmpnn/

## Citations

- Dauparas, J., Anishchenko, I., Bennett, N., Bai, H., Ragotte, R. J.,
  Milles, L. F., Wicky, B. I. M., Courbet, A., de Haas, R. J., Bethel, N.,
  Leung, P. J. Y., Huddy, T. F., Pellock, S., Tischer, D., Chan, F.,
  Koepnick, B., Nguyen, H., Kang, A., Sankaran, B., Bera, A. K.,
  King, N. P. & Baker, D.
  *Robust deep learning–based protein sequence design using ProteinMPNN.*
  Science 378, 49–56 (2022).

- Dauparas, J., Lee, G. R., Pecoraro, R., An, L., Anishchenko, I.,
  Glasscock, C. & Baker, D.
  *Atomic context-conditioned protein sequence design using LigandMPNN.*
  Nature Methods (2025).

## License

Both upstream repositories are MIT-licensed. Redistribution is permitted; this
NOTICES.md file preserves attribution to the upstream authors. See the
respective repositories for the full license text.

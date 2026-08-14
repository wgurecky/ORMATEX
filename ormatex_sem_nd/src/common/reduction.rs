/// Mapping from full function-space DOFs to a reduced system.
pub(crate) struct ReducedDofMap {
    /// Full DOF -> reduced DOF. `Some(i)` retains a DOF at reduced index `i`;
    /// `None` eliminates it. Multiple full DOFs may share one reduced index
    /// when periodic DOFs are identified.
    dof_lut: Vec<Option<usize>>,
    prescribed_values: Vec<Option<f64>>,
    n_reduced: usize,
}

impl ReducedDofMap {
    /// Create a map that retains every full DOF in its original order.
    pub(crate) fn identity(full_dof_count: usize) -> Self {
        Self {
            dof_lut: (0..full_dof_count).map(Some).collect(),
            prescribed_values: vec![None; full_dof_count],
            n_reduced: full_dof_count,
        }
    }

    /// Create a map from each full DOF to its representative full DOF.
    ///
    /// Representatives are assigned reduced indices in first-seen order, so
    /// equivalent full DOFs share the same reduced index.
    pub(crate) fn from_representatives(representatives: Vec<usize>) -> Self {
        let full_dof_count = representatives.len();
        let mut representatives_in_order = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &representative in &representatives {
            assert!(
                representative < full_dof_count,
                "DOF representative out of range"
            );
            if seen.insert(representative) {
                representatives_in_order.push(representative);
            }
        }
        let mut reduced_index = vec![usize::MAX; full_dof_count];
        for (reduced, representative) in representatives_in_order.iter().enumerate() {
            reduced_index[*representative] = reduced;
        }
        let dof_lut = representatives
            .into_iter()
            .map(|representative| Some(reduced_index[representative]))
            .collect();
        Self::new(
            dof_lut,
            vec![None; full_dof_count],
            representatives_in_order.len(),
        )
    }

    /// Create a map that eliminates the supplied full DOFs and prescribes
    /// their values.
    ///
    /// Repeated entries for one DOF are accepted only when their values agree
    /// within a relative tolerance.
    pub(crate) fn from_dirichlet_values<I>(full_dof_count: usize, values: I) -> Self
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let mut prescribed_values: Vec<Option<f64>> = vec![None; full_dof_count];
        for (dof, value) in values {
            assert!(dof < full_dof_count, "Dirichlet DOF out of range");
            assert!(value.is_finite(), "Dirichlet value must be finite");
            if let Some(previous) = prescribed_values[dof] {
                assert!(
                    (previous - value).abs() <= 1e-12 * previous.abs().max(value.abs()).max(1.0),
                    "conflicting Dirichlet values for DOF {dof}"
                );
            } else {
                prescribed_values[dof] = Some(value);
            }
        }
        let mut reduced = 0;
        let dof_lut = prescribed_values
            .iter()
            .map(|value| {
                if value.is_some() {
                    None
                } else {
                    let index = Some(reduced);
                    reduced += 1;
                    index
                }
            })
            .collect();
        Self::new(dof_lut, prescribed_values, reduced)
    }

    fn new(
        dof_lut: Vec<Option<usize>>,
        prescribed_values: Vec<Option<f64>>,
        n_reduced: usize,
    ) -> Self {
        assert_eq!(
            dof_lut.len(),
            prescribed_values.len(),
            "DOF map and prescribed-value lengths must match"
        );
        let mut present = vec![false; n_reduced];
        for (full, &reduced) in dof_lut.iter().enumerate() {
            match reduced {
                Some(reduced) => {
                    assert!(reduced < n_reduced, "reduced DOF out of range");
                    assert!(
                        prescribed_values[full].is_none(),
                        "retained DOF cannot have a prescribed value"
                    );
                    present[reduced] = true;
                }
                None => assert!(
                    prescribed_values[full].is_some(),
                    "eliminated DOF must have a prescribed value"
                ),
            }
        }
        assert!(
            present.into_iter().all(|present| present),
            "reduced DOF indices must be contiguous"
        );
        Self {
            dof_lut,
            prescribed_values,
            n_reduced,
        }
    }

    /// Return the reduced target for `full_dof`, or `None` if it is eliminated.
    pub(crate) fn target(&self, full_dof: usize) -> Option<usize> {
        self.dof_lut[full_dof]
    }

    /// Return the number of full-space DOFs represented by this map.
    pub(crate) fn full_size(&self) -> usize {
        self.dof_lut.len()
    }

    /// Return the number of retained reduced DOFs.
    pub(crate) fn reduced_size(&self) -> usize {
        self.n_reduced
    }

    /// Map each cell's full local DOFs to reduced DOFs and prescribed values.
    pub(crate) fn map_cells(
        &self,
        cell_dofs: &[Vec<usize>],
    ) -> (Vec<Vec<Option<usize>>>, Vec<Vec<Option<f64>>>) {
        let reduced = cell_dofs
            .iter()
            .map(|dofs| dofs.iter().map(|&dof| self.target(dof)).collect())
            .collect();
        let prescribed = cell_dofs
            .iter()
            .map(|dofs| {
                dofs.iter()
                    .map(|&dof| self.prescribed_values[dof])
                    .collect()
            })
            .collect();
        (reduced, prescribed)
    }
}

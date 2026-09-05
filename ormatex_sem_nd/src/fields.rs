use std::collections::HashMap;

/// Resolved local-to-global field bindings for one kernel term.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FieldSelection {
    /// Global field IDs used to build the compact kernel state.
    pub(crate) inputs: Vec<usize>,
    /// Global field IDs receiving the compact kernel residual.
    pub(crate) outputs: Vec<usize>,
}

/// Ordered names for the scalar solution fields in a SEM problem.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldRegistry {
    names: Vec<String>,
    ids: HashMap<String, usize>,
}

impl FieldRegistry {
    /// Build a registry from field names in their system-vector order.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        assert!(
            !names.is_empty(),
            "a problem must contain at least one field"
        );
        let mut ids = HashMap::with_capacity(names.len());
        for (id, name) in names.iter().enumerate() {
            assert!(!name.is_empty(), "field names must not be empty");
            assert!(
                ids.insert(name.clone(), id).is_none(),
                "duplicate field name: {name}"
            );
        }
        Self { names, ids }
    }

    /// Number of registered fields.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the registry contains no fields.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Field names in system-vector order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Return the name at a numeric field ID.
    pub fn name(&self, id: usize) -> Option<&str> {
        self.names.get(id).map(String::as_str)
    }

    /// Return the numeric field ID for `name`.
    pub fn id(&self, name: &str) -> Option<usize> {
        self.ids.get(name).copied()
    }

    pub(crate) fn resolve_selection(
        &self,
        input_count: usize,
        input_names: Option<Vec<String>>,
        output_count: usize,
        output_names: Option<Vec<String>>,
        context: &str,
    ) -> FieldSelection {
        FieldSelection {
            inputs: self.resolve_names(input_count, input_names, context, "input"),
            outputs: self.resolve_names(output_count, output_names, context, "output"),
        }
    }

    fn resolve_names(
        &self,
        count: usize,
        names: Option<Vec<String>>,
        context: &str,
        kind: &str,
    ) -> Vec<usize> {
        assert!(
            count > 0,
            "{context} must contain at least one {kind} field"
        );
        match names {
            Some(names) => {
                assert_eq!(
                    names.len(),
                    count,
                    "{context} {kind} field-name count does not match field count"
                );
                let mut ids = Vec::with_capacity(count);
                for name in names {
                    let id = self
                        .id(&name)
                        .unwrap_or_else(|| panic!("{context} references unknown field: {name}"));
                    assert!(
                        !ids.contains(&id),
                        "{context} {kind} fields contain duplicate name: {name}"
                    );
                    ids.push(id);
                }
                ids
            }
            None => {
                assert_eq!(
                    count,
                    self.len(),
                    "{context} {kind} field count does not match the SEM problem"
                );
                (0..count).collect()
            }
        }
    }
}

/// Values and representative positions for one reduced solution field.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldValues<P> {
    pub positions: Vec<P>,
    pub values: Vec<f64>,
}

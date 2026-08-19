use std::collections::HashMap;

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
}

/// Values and representative positions for one reduced solution field.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldValues<P> {
    pub positions: Vec<P>,
    pub values: Vec<f64>,
}

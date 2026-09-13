//! Facet-term containers selecting boundary kernels per facet.
use super::boundary_traits::{StateBoundaryIntegrator, StateTensorBoundaryIntegrator};
use std::collections::HashMap;
use std::sync::Arc;

/// Immutable state-dependent boundary terms selected by mesh entity index.
#[derive(Clone, Default)]
pub struct StateBoundaryTerms {
    default: Option<Arc<dyn StateBoundaryIntegrator>>,
    overrides: HashMap<usize, Arc<dyn StateBoundaryIntegrator>>,
}

impl StateBoundaryTerms {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default<K>(mut self, kernel: K) -> Self
    where
        K: StateBoundaryIntegrator + 'static,
    {
        self.default = Some(Arc::new(kernel));
        self
    }

    pub fn with_entities<K, I>(mut self, entities: I, kernel: K) -> Self
    where
        K: StateBoundaryIntegrator + 'static,
        I: IntoIterator<Item = usize>,
    {
        let kernel: Arc<dyn StateBoundaryIntegrator> = Arc::new(kernel);
        for entity in entities {
            assert!(
                self.overrides.insert(entity, Arc::clone(&kernel)).is_none(),
                "state boundary entity configured more than once"
            );
        }
        self
    }

    pub(crate) fn kernel_for(&self, entity: usize) -> Option<&dyn StateBoundaryIntegrator> {
        self.overrides
            .get(&entity)
            .or(self.default.as_ref())
            .map(AsRef::as_ref)
    }
}

/// Immutable tensor boundary terms selected by mesh entity index.
///
/// This is deliberately separate from [`StateBoundaryTerms`]: weak boundary
/// kernels remain usable without also implementing a tensor boundary kernel.
#[derive(Clone, Default)]
pub struct StateTensorBoundaryTerms<const GDIM: usize> {
    default: Option<Arc<dyn StateTensorBoundaryIntegrator<GDIM>>>,
    overrides: HashMap<usize, Arc<dyn StateTensorBoundaryIntegrator<GDIM>>>,
}

impl<const GDIM: usize> StateTensorBoundaryTerms<GDIM> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no boundary kernel is configured at all. Hot assembly paths
    /// use this to skip the facet loop entirely.
    pub fn is_empty(&self) -> bool {
        self.default.is_none() && self.overrides.is_empty()
    }

    pub fn with_default<K>(mut self, kernel: K) -> Self
    where
        K: StateTensorBoundaryIntegrator<GDIM> + 'static,
    {
        self.default = Some(Arc::new(kernel));
        self
    }

    pub fn with_entities<K, I>(mut self, entities: I, kernel: K) -> Self
    where
        K: StateTensorBoundaryIntegrator<GDIM> + 'static,
        I: IntoIterator<Item = usize>,
    {
        let kernel: Arc<dyn StateTensorBoundaryIntegrator<GDIM>> = Arc::new(kernel);
        for entity in entities {
            assert!(
                self.overrides.insert(entity, Arc::clone(&kernel)).is_none(),
                "tensor state boundary entity configured more than once"
            );
        }
        self
    }

    pub(crate) fn kernel_for(
        &self,
        entity: usize,
    ) -> Option<&dyn StateTensorBoundaryIntegrator<GDIM>> {
        self.overrides
            .get(&entity)
            .or(self.default.as_ref())
            .map(AsRef::as_ref)
    }
}

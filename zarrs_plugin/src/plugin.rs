use crate::PluginCreateError;

/// A plugin.
pub struct Plugin<TPlugin, TInput> {
    /// Tests if the name is a match for this plugin.
    match_name_fn: fn(name: &str) -> bool,
    /// Create an implementation of this plugin from metadata.
    create_fn: fn(input: &TInput) -> Result<TPlugin, PluginCreateError>,
}

/// A plugin (two parameters).
pub struct Plugin2<TPlugin, TInput1, TInput2> {
    /// Tests if the name is a match for this plugin.
    match_name_fn: fn(name: &str) -> bool,
    /// Create an implementation of this plugin from metadata.
    create_fn: fn(input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError>,
}

impl<TPlugin, TInput> Plugin<TPlugin, TInput> {
    /// Create a new plugin for registration.
    pub const fn new(
        match_name_fn: fn(name: &str) -> bool,
        create_fn: fn(inputs: &TInput) -> Result<TPlugin, PluginCreateError>,
    ) -> Self {
        Self {
            match_name_fn,
            create_fn,
        }
    }

    /// Create a `TPlugin` plugin from `inputs`.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if plugin creation fails due to either:
    ///  - metadata name being unregistered,
    ///  - or the configuration is invalid, or
    ///  - some other reason specific to the plugin.
    pub fn create(&self, input: &TInput) -> Result<TPlugin, PluginCreateError> {
        (self.create_fn)(input)
    }

    /// Returns true if this plugin is associated with `name`.
    #[must_use]
    pub fn match_name(&self, name: &str) -> bool {
        (self.match_name_fn)(name)
    }
}

impl<TPlugin, TInput1, TInput2> Plugin2<TPlugin, TInput1, TInput2> {
    /// Create a new plugin for registration.
    pub const fn new(
        match_name_fn: fn(name: &str) -> bool,
        create_fn: fn(input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError>,
    ) -> Self {
        Self {
            match_name_fn,
            create_fn,
        }
    }

    /// Create a `TPlugin` plugin from `inputs`.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if plugin creation fails due to either:
    ///  - metadata name being unregistered,
    ///  - or the configuration is invalid, or
    ///  - some other reason specific to the plugin.
    pub fn create(&self, input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError> {
        (self.create_fn)(input1, input2)
    }

    /// Returns true if this plugin is associated with `name`.
    #[must_use]
    pub fn match_name(&self, name: &str) -> bool {
        (self.match_name_fn)(name)
    }
}

const SOURCE_COLORS = {
  model_sensitivity: 'var(--blue)',
  model_importance: 'var(--blue)',
  correlation_discovery: 'var(--teal)',
  cross_correlation: 'var(--teal)',
  recipe_limit: 'var(--amber)',
}

const SOURCE_LABELS = {
  model_sensitivity: 'model sensitivity',
  model_importance: 'model importance',
  correlation_discovery: 'correlation',
  cross_correlation: 'cross-correlation',
  recipe_limit: 'recipe limit',
}

export const sourceColor = (source) => SOURCE_COLORS[source] || 'var(--muted)'
export const sourceLabel = (source) => SOURCE_LABELS[source] || source
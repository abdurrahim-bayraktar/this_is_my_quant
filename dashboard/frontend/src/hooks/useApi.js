const API_BASE = 'http://localhost:8000/api';

export async function fetchApi(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function useApi(path) {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(!!path);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    if (!path) {
      setData(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchApi(path)
      .then(d => { if (!cancelled) setData(d); })
      .catch(e => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [path]);

  return { data, loading, error };
}

import React from 'react';

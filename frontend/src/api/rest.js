const BASE = "/api";

async function request(path, options = {}) {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp;
}

// -- Pipeline --

export async function startPipeline() {
  return (await request("/pipeline/start", { method: "POST" })).json();
}

export async function stopPipeline() {
  return (await request("/pipeline/stop", { method: "POST" })).json();
}

// -- Channels --

export async function getChannels() {
  return (await request("/channels")).json();
}

export async function getChannel(channelId) {
  return (await request(`/channel/${channelId}`)).json();
}

export async function addChannel(source) {
  return (
    await request("/channel/add", {
      method: "POST",
      body: JSON.stringify({ source }),
    })
  ).json();
}

export async function removeChannel(channelId) {
  return (
    await request("/channel/remove", {
      method: "POST",
      body: JSON.stringify({ channel_id: channelId }),
    })
  ).json();
}

export async function setChannelPhase(channelId, phase) {
  return (
    await request(`/channel/${channelId}/phase`, {
      method: "POST",
      body: JSON.stringify({ phase }),
    })
  ).json();
}

// -- Config --

export async function updateConfig(params) {
  return (
    await request("/config", {
      method: "PATCH",
      body: JSON.stringify(params),
    })
  ).json();
}

// -- Alerts --

export async function getAlerts({ limit, type, channel } = {}) {
  const params = new URLSearchParams();
  if (limit) params.set("limit", limit);
  if (type) params.set("type", type);
  if (channel !== undefined) params.set("channel", channel);
  const qs = params.toString();
  return (await request(`/alerts${qs ? `?${qs}` : ""}`)).json();
}

export async function getAlert(alertId) {
  return (await request(`/alert/${alertId}`)).json();
}

// -- Replay --

export function replayUrl(alertId) {
  return `/api/alert/${alertId}/replay`;
}

export async function getReplayStatus(alertId) {
  const resp = await fetch(`${BASE}/alert/${alertId}/replay`);
  if (resp.status === 202) return { status: (await resp.json()).status };
  if (resp.status === 200) return { status: "ready" };
  throw new Error(`Replay error: ${resp.status}`);
}

// -- Snapshots --

export function snapshotUrl(trackId) {
  return `/snapshot/${trackId}`;
}

// -- Site config --

export async function saveSiteConfig(config) {
  return (
    await request("/site/config", {
      method: "POST",
      body: JSON.stringify(config),
    })
  ).json();
}

export async function loadSiteConfig(siteId) {
  return (await request(`/site/config?site_id=${siteId}`)).json();
}

export async function listSiteConfigs() {
  return (await request("/site/configs")).json();
}

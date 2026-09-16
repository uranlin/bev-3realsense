const $ = (id) => document.getElementById(id);
let mode = "demo",
  ready = false;
async function api(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const d = await r.json();
  if (!r.ok)
    throw Error(
      typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail),
    );
  return d;
}
function show(d) {
  if (d.bev_image) $("color").src = d.bev_image;
  if (d.bev_freespace) $("free").src = d.bev_freespace;
}
async function busy(fn) {
  $("capture").disabled = $("plan").disabled = true;
  try {
    await fn();
  } catch (e) {
    $("status").textContent = e.message;
  } finally {
    $("capture").disabled = false;
    $("plan").disabled = !ready;
  }
}
$("capture").onclick = () =>
  busy(async () => {
    let route = "/demo";
    if (mode !== "demo") {
      const r = await fetch("/cameras");
      const d = await r.json();
      route = d.multi_rig_active ? "/infer_multi" : "/infer_realsense";
    }
    const d = await api(route);
    show(d);
    ready = true;
    $("status").textContent = "觀測已更新，可設定目標並規劃。";
  });
$("plan").onclick = () =>
  busy(async () => {
    const d = await api("/plan", {
      goal_x: Number($("x").value),
      goal_z: Number($("z").value),
      robot_radius: Number($("radius").value),
    });
    show(d);
    $("status").textContent = d.success
      ? `路徑長度 ${d.path_length_m} m\n${d.n_waypoints} 個路徑點`
      : "找不到路徑：請確認目標位於已觀測的可通行區域。";
  });
fetch("/health")
  .then((r) => r.json())
  .then((d) => {
    mode = d.mode;
    $("mode").textContent =
      mode === "demo" ? "離線示範" : "RealSense / " + mode;
    $("source").textContent =
      mode === "demo"
        ? "三個合成 BEV 觀測區域；用來驗證融合與規劃，不代表實機相機結果。"
        : "擷取已連接相機的單次觀測。";
    if (mode === "demo") $("capture").click();
  })
  .catch((e) => {
    $("status").textContent = "無法連接服務：" + e.message;
  });

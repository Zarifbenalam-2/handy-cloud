import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands } from "@/bindings";
import { useSettings } from "../../../hooks/useSettings";
import { useSettingsStore } from "../../../stores/settingsStore";
import { SettingsGroup } from "../../ui/SettingsGroup";

type CloudProviderChoice = "off" | "perplexity" | "gemini" | "soniox";

const inputClass =
  "w-full rounded-md border border-line bg-transparent px-3 py-2 text-sm outline-none focus:border-neutral-400";

export const CloudEngineCard: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting } = useSettings();
  const refreshSettings = useSettingsStore((s) => s.refreshSettings);

  const cloud = getSetting("cloud_transcription") ?? null;
  const [provider, setProvider] = useState<CloudProviderChoice>("off");
  const [geminiKey, setGeminiKey] = useState("");
  const [sonioxKey, setSonioxKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setProvider((cloud?.provider as CloudProviderChoice) ?? "off");
    setGeminiKey(cloud?.gemini_api_key ?? "");
    setSonioxKey(cloud?.soniox_api_key ?? "");
  }, [cloud]);

  const save = async () => {
    setSaving(true);
    setMessage(null);
    try {
      const result = await commands.setCloudTranscriptionSetting({
        provider: provider === "off" ? null : provider,
        geminiApiKey: geminiKey || null,
        sonioxApiKey: sonioxKey || null,
      });
      if (result.status === "error") {
        throw new Error(result.error);
      }
      await refreshSettings();
      setMessage(t("settings.cloudEngine.saved"));
    } catch (error) {
      setMessage(
        `${t("settings.cloudEngine.saveFailed")}: ${String(error)}`,
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <SettingsGroup title={t("settings.cloudEngine.title")}>
      <div className="space-y-3 p-3">
        <p className="text-sm text-neutral-500">
          {t("settings.cloudEngine.description")}
        </p>
        <div className="flex items-center gap-3">
          <span className="text-sm">
            {t("settings.cloudEngine.provider")}
          </span>
          <select
            className="rounded-md border border-line bg-transparent px-2 py-1.5 text-sm"
            value={provider}
            onChange={(e) => setProvider(e.target.value as CloudProviderChoice)}
          >
            <option value="off">{t("settings.cloudEngine.off")}</option>
            <option value="perplexity">
              {t("settings.cloudEngine.perplexity")}
            </option>
            <option value="gemini">{t("settings.cloudEngine.gemini")}</option>
            <option value="soniox">{t("settings.cloudEngine.soniox")}</option>
          </select>
        </div>
        {provider === "gemini" && (
          <div className="space-y-1">
            <label className="text-sm" htmlFor="cloud-gemini-key">
              {t("settings.cloudEngine.geminiKey")}
            </label>
            <input
              id="cloud-gemini-key"
              type="password"
              className={inputClass}
              value={geminiKey}
              onChange={(e) => setGeminiKey(e.target.value)}
              placeholder={t("settings.cloudEngine.geminiKeyHint")}
            />
          </div>
        )}
        {provider === "soniox" && (
          <div className="space-y-1">
            <label className="text-sm" htmlFor="cloud-soniox-key">
              {t("settings.cloudEngine.sonioxKey")}
            </label>
            <input
              id="cloud-soniox-key"
              type="password"
              className={inputClass}
              value={sonioxKey}
              onChange={(e) => setSonioxKey(e.target.value)}
            />
          </div>
        )}
        <div className="flex items-center gap-3">
          <button
            className="rounded-md bg-white/10 px-3 py-1.5 text-sm hover:bg-white/20 disabled:opacity-50"
            onClick={save}
            disabled={saving}
          >
            {saving ? "…" : t("settings.cloudEngine.save")}
          </button>
          {message && (
            <span className="text-sm text-neutral-500">{message}</span>
          )}
        </div>
      </div>
    </SettingsGroup>
  );
};

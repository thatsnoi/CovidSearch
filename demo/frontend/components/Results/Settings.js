import { useState, useEffect } from 'react'

export default function Settings({ updateSettings }) {
  const [settings, setSettings] = useState({
    ce: false,
    top_k: 20,
    fuse: false,
  })

  const handleChange = (event) => {
    setSettings({
      ...settings,
      [event.target.name]: event.target.value,
    })
  }

  const handleChangeRadioButton = (event) => {
    setSettings({
      ...settings,
      [event.target.name]: !settings[event.target.name],
    })
  }

  useEffect(() => {
    updateSettings(settings)
  }, [settings, updateSettings])

  return (
    <div className="space-y-5">
      <div>
        <label>Enable Cross-Encoder?</label>
        <fieldset className="flex space-x-4" defaultValue="true">
          <div>
            <input
              type="radio"
              id="ceTrue"
              name="ce"
              value="true"
              checked={settings.ce}
              onChange={handleChangeRadioButton}
            />
            <label className="pl-2" htmlFor="cdTrue">
              True
            </label>
          </div>
          <div>
            <input
              type="radio"
              id="ceFalse"
              name="ce"
              value="false"
              checked={!settings.ce}
              onChange={handleChangeRadioButton}
            />
            <label className="pl-2" htmlFor="ceFalse">
              False
            </label>
          </div>
        </fieldset>
      </div>
      <div className="flex flex-col">
        {settings.ce && (
          <>
            <label>Cross-Encoder Top-k:</label>
            <input
              className="rounded-full border-2 border-indigo-200 px-4 w-32"
              value={settings.top_k}
              name="top_k"
              onChange={handleChange}
              type="number"
            ></input>
          </>
        )}
      </div>
      <div>
        <label>Fuse with BM25?</label>
        <fieldset className="flex space-x-4" defaultValue="true">
          <div>
            <input
              type="radio"
              id="fuseTrue"
              name="fuse"
              value="true"
              checked={settings.fuse}
              onChange={handleChangeRadioButton}
            />
            <label className="pl-2" htmlFor="fuseTrue">
              True
            </label>
          </div>
          <div>
            <input
              type="radio"
              id="fuseFalse"
              name="fuse"
              checked={!settings.fuse}
              onChange={handleChangeRadioButton}
            />
            <label className="pl-2" htmlFor="fuseFalse">
              False
            </label>
          </div>
        </fieldset>
      </div>
    </div>
  )
}

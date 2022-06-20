import SearchInput from '../components/Search/SearchInput'
import { FiGithub } from 'react-icons/fi'
import Result from '../components/Results/Result'

export default function Results() {
  return (
    <div className="flex flex-col relative h-screen">
      <div className="flex justify-between items-center p-4 bg-indigo-900 h-20 w-full">
        <div className="w-2/3 flex items-center space-x-10">
          <h1 className="text-white font-sans text-3xl font-semibold">
            Covid<span className=" text-indigo-200">Search</span>
          </h1>
          <SearchInput />
        </div>
        <FiGithub className="text-white text-2xl cursor-pointer" />
      </div>
      <div className="flex h-full">
        <div className="h-full bg-gray-200 p-5 whitespace-nowrap">
          <h2 className="text-2xl font-semibold pb-5">Settings</h2>
          <div className="space-y-5">
            <div className="flex flex-col">
              <label>Cross-Encoder Top-k:</label>
              <input
                className="rounded-full border-2 border-indigo-200 px-4 w-32"
                defaultValue={20}
                type="number"
              ></input>
            </div>
            <div>
              <label>Fuse with BM25?</label>
              <fieldset className="flex space-x-4" defaultValue="true">
                <div>
                  <input type="radio" id="fuseTrue" name="fuse" value="true" />
                  <label className="pl-2" htmlFor="fuseTrue">
                    True
                  </label>
                </div>
                <div>
                  <input
                    type="radio"
                    id="fuseFalse"
                    name="fuse"
                    value="false"
                  />
                  <label className="pl-2" htmlFor="fuseFalse">
                    False
                  </label>
                </div>
              </fieldset>
            </div>
          </div>
        </div>
        <div className="px-10 py-5 space-y-5">
          <Result />
          <Result />
          <Result />
        </div>
      </div>
    </div>
  )
}

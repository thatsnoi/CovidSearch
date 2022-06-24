export default function Result({ title, text }) {
  return (
    <div className="flex flex-col">
      <h2 className="font-semibold text-xl cursor-pointer hover:underline">
        {title}
      </h2>
      <p className="font-semibold text-indigo-900">
        MR Ghorab, D Zhou, A O connor, V Wade
      </p>
      <p className="text-sm line-clamp-4">{text}</p>
    </div>
  )
}

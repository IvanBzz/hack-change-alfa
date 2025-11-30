import React from 'react';
import { formatFeatureName } from '../utils';

const ParamInput = React.memo(({ paramKey, value, onChange }) => {
  return (
    <div className="flex flex-col gap-1 p-2 hover:bg-gray-50 rounded-lg transition-colors">
       <label className="text-xs font-bold text-gray-500 truncate uppercase tracking-wide" title={formatFeatureName(paramKey)}>
         {formatFeatureName(paramKey)}
       </label>
       {typeof value === 'number' ? (
          <input 
            type="number" 
            className="p-2 border border-gray-200 rounded-lg text-sm focus:border-red-500 outline-none bg-white text-gray-900 placeholder-gray-400"
            value={value}
            onChange={(e) => onChange(paramKey, parseFloat(e.target.value))}
          />
       ) : (
          <input 
            type="text" 
            className="p-2 border border-gray-200 rounded-lg text-sm focus:border-red-500 outline-none bg-white text-gray-900 placeholder-gray-400"
            value={value}
            onChange={(e) => onChange(paramKey, e.target.value)}
          />
       )}
    </div>
  );
}, (prevProps, nextProps) => {
  return prevProps.value === nextProps.value && prevProps.paramKey === nextProps.paramKey;
});

export default ParamInput;
